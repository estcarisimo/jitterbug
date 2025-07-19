"""
API routes for Jitterbug REST API.
"""

import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import traceback
import logging

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse

from .models import (
    AnalysisRequestAPI, AnalysisResponseAPI, AnalysisResultAPI,
    ValidationRequestAPI, ValidationResponseAPI,
    AlgorithmComparisonRequestAPI, AlgorithmComparisonResponseAPI,
    HealthCheckAPI, StatusAPI, ErrorResponseAPI,
    RTTMeasurementAPI, RTTDatasetAPI, CongestionInferenceAPI, ChangePointAPI,
    convert_to_api_model, convert_from_api_model
)
from ..analyzer import JitterbugAnalyzer
from ..models import JitterbugConfig, RTTMeasurement, RTTDataset, MinimumRTTDataset
from ..io import DataLoader
from ..detection import ChangePointDetector


# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global state
service_stats = {
    'requests_processed': 0,
    'errors': 0,
    'start_time': time.time(),
    'response_times': []
}


class RequestTracker:
    """Track requests for monitoring."""
    
    def __init__(self):
        self.start_time = time.time()
    
    def record_request(self, response_time: float, success: bool = True):
        service_stats['requests_processed'] += 1
        service_stats['response_times'].append(response_time)
        
        # Keep only last 100 response times
        if len(service_stats['response_times']) > 100:
            service_stats['response_times'].pop(0)
        
        if not success:
            service_stats['errors'] += 1


# Dependency for request tracking
def get_request_tracker():
    return RequestTracker()


@router.get("/health", response_model=HealthCheckAPI)
async def health_check():
    """Health check endpoint."""
    try:
        # Check dependencies
        dependencies = {}
        
        try:
            import numpy
            dependencies['numpy'] = "available"
        except ImportError:
            dependencies['numpy'] = "missing"
        
        try:
            import pandas
            dependencies['pandas'] = "available"
        except ImportError:
            dependencies['pandas'] = "missing"
        
        try:
            import ruptures
            dependencies['ruptures'] = "available"
        except ImportError:
            dependencies['ruptures'] = "missing"
        
        try:
            import torch
            dependencies['torch'] = "available"
        except ImportError:
            dependencies['torch'] = "missing"
        
        return HealthCheckAPI(
            status="healthy",
            timestamp=datetime.now(),
            version="2.0.0",
            uptime=time.time() - service_stats['start_time'],
            dependencies=dependencies
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/status", response_model=StatusAPI)
async def get_status():
    """Get service status and statistics."""
    try:
        avg_response_time = (
            sum(service_stats['response_times']) / len(service_stats['response_times'])
            if service_stats['response_times'] else 0
        )
        
        return StatusAPI(
            service="jitterbug-api",
            status="running",
            timestamp=datetime.now(),
            requests_processed=service_stats['requests_processed'],
            errors=service_stats['errors'],
            avg_response_time=avg_response_time
        )
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")


@router.post("/analyze", response_model=AnalysisResponseAPI)
async def analyze_data(
    request: AnalysisRequestAPI,
    background_tasks: BackgroundTasks,
    tracker: RequestTracker = Depends(get_request_tracker)
):
    """Analyze RTT data for network congestion inference."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info(f"Starting analysis request {request_id}")
        
        # Convert API models to core models
        core_measurements = [
            convert_from_api_model(measurement) 
            for measurement in request.data.measurements
        ]
        
        # Create RTT dataset
        rtt_dataset = RTTDataset(measurements=core_measurements)
        
        # Create configuration
        config = JitterbugConfig()
        if request.config:
            config = config.copy(update=request.config)
        
        # Override with request parameters
        if request.algorithm:
            config.change_point_detection.algorithm = request.algorithm
        if request.method:
            config.jitter_analysis.method = request.method
        if request.threshold:
            config.change_point_detection.threshold = request.threshold
        if request.min_time_elapsed:
            config.change_point_detection.min_time_elapsed = request.min_time_elapsed
        
        # Create analyzer
        analyzer = JitterbugAnalyzer(config)
        
        # Perform analysis
        logger.info(f"Analyzing {len(core_measurements)} measurements")
        results = analyzer.analyze_dataset(rtt_dataset)
        
        # Detect change points
        change_points = []
        if results.inferences:
            # Create MinimumRTTDataset for change point detection
            min_rtt_data = MinimumRTTDataset(
                measurements=core_measurements,
                interval_minutes=15  # Default interval
            )
            
            detector = ChangePointDetector(config.change_point_detection)
            change_points = detector.detect(min_rtt_data)
        
        # Convert results to API models
        api_inferences = [convert_to_api_model(inf) for inf in results.inferences]
        api_change_points = [convert_to_api_model(cp) for cp in change_points]
        
        analysis_result = AnalysisResultAPI(
            inferences=api_inferences,
            change_points=api_change_points,
            metadata={
                'total_measurements': len(core_measurements),
                'total_periods': len(results.inferences),
                'congested_periods': len(results.get_congested_periods()),
                'change_points': len(change_points),
                'algorithm_used': request.algorithm or 'ruptures',
                'method_used': request.method or 'jitter_dispersion',
                'analysis_duration': time.time() - start_time,
                'request_id': request_id
            }
        )
        
        execution_time = time.time() - start_time
        
        # Record successful request
        background_tasks.add_task(tracker.record_request, execution_time, True)
        
        logger.info(f"Analysis completed successfully in {execution_time:.2f}s")
        
        return AnalysisResponseAPI(
            success=True,
            result=analysis_result,
            error=None,
            execution_time=execution_time,
            request_id=request_id
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = str(e)
        
        logger.error(f"Analysis failed: {error_msg}")
        logger.error(traceback.format_exc())
        
        # Record failed request
        background_tasks.add_task(tracker.record_request, execution_time, False)
        
        return AnalysisResponseAPI(
            success=False,
            result=None,
            error=error_msg,
            execution_time=execution_time,
            request_id=request_id
        )


@router.post("/validate", response_model=ValidationResponseAPI)
async def validate_data(
    request: ValidationRequestAPI,
    background_tasks: BackgroundTasks,
    tracker: RequestTracker = Depends(get_request_tracker)
):
    """Validate RTT data quality and format."""
    start_time = time.time()
    
    try:
        logger.info("Starting data validation")
        
        # Convert API models to core models
        core_measurements = [
            convert_from_api_model(measurement) 
            for measurement in request.data.measurements
        ]
        
        # Create RTT dataset
        rtt_dataset = RTTDataset(measurements=core_measurements)
        
        # Validate data
        data_loader = DataLoader()
        validation_results = data_loader.validate_data(rtt_dataset)
        
        execution_time = time.time() - start_time
        
        # Record request
        background_tasks.add_task(tracker.record_request, execution_time, True)
        
        logger.info(f"Data validation completed in {execution_time:.2f}s")
        
        return ValidationResponseAPI(
            valid=validation_results['valid'],
            errors=validation_results.get('errors', []),
            warnings=validation_results.get('warnings', []),
            metrics=validation_results.get('metrics', {})
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = str(e)
        
        logger.error(f"Data validation failed: {error_msg}")
        
        # Record failed request
        background_tasks.add_task(tracker.record_request, execution_time, False)
        
        return ValidationResponseAPI(
            valid=False,
            errors=[error_msg],
            warnings=[],
            metrics={}
        )


@router.post("/compare-algorithms", response_model=AlgorithmComparisonResponseAPI)
async def compare_algorithms(
    request: AlgorithmComparisonRequestAPI,
    background_tasks: BackgroundTasks,
    tracker: RequestTracker = Depends(get_request_tracker)
):
    """Compare different change point detection algorithms."""
    start_time = time.time()
    
    try:
        logger.info(f"Starting algorithm comparison: {request.algorithms}")
        
        # Convert API models to core models
        core_measurements = [
            convert_from_api_model(measurement) 
            for measurement in request.data.measurements
        ]
        
        # Create MinimumRTTDataset
        min_rtt_data = MinimumRTTDataset(
            measurements=core_measurements,
            interval_minutes=15
        )
        
        # Compare algorithms
        results = {}
        performance = {}
        
        for algorithm in request.algorithms:
            try:
                # Create configuration
                config = JitterbugConfig()
                if request.config:
                    config = config.copy(update=request.config)
                
                config.change_point_detection.algorithm = algorithm
                
                # Detect change points
                detector = ChangePointDetector(config.change_point_detection)
                
                algo_start = time.time()
                change_points = detector.detect(min_rtt_data)
                algo_time = time.time() - algo_start
                
                # Convert to API models
                api_change_points = [convert_to_api_model(cp) for cp in change_points]
                
                results[algorithm] = api_change_points
                performance[algorithm] = algo_time
                
                logger.info(f"Algorithm {algorithm}: {len(change_points)} change points in {algo_time:.2f}s")
                
            except Exception as e:
                logger.warning(f"Algorithm {algorithm} failed: {e}")
                results[algorithm] = []
                performance[algorithm] = float('inf')
        
        execution_time = time.time() - start_time
        
        # Record request
        background_tasks.add_task(tracker.record_request, execution_time, True)
        
        logger.info(f"Algorithm comparison completed in {execution_time:.2f}s")
        
        return AlgorithmComparisonResponseAPI(
            success=True,
            results=results,
            performance=performance,
            error=None,
            execution_time=execution_time
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = str(e)
        
        logger.error(f"Algorithm comparison failed: {error_msg}")
        logger.error(traceback.format_exc())
        
        # Record failed request
        background_tasks.add_task(tracker.record_request, execution_time, False)
        
        return AlgorithmComparisonResponseAPI(
            success=False,
            results=None,
            performance=None,
            error=error_msg,
            execution_time=execution_time
        )


@router.get("/algorithms")
async def get_available_algorithms():
    """Get list of available change point detection algorithms."""
    try:
        algorithms = {
            'ruptures': {
                'name': 'Ruptures',
                'description': 'Fast change point detection using various kernels',
                'available': True,
                'models': ['rbf', 'l1', 'l2', 'normal']
            },
            'bcp': {
                'name': 'Bayesian Change Point',
                'description': 'Probabilistic change point detection',
                'available': True,
                'models': ['student_t']
            },
            'torch': {
                'name': 'PyTorch Neural Network',
                'description': 'Deep learning-based change point detection',
                'available': True,
                'models': ['cnn_lstm']
            }
        }
        
        # Check actual availability
        try:
            import ruptures
        except ImportError:
            algorithms['ruptures']['available'] = False
        
        try:
            import torch
        except ImportError:
            algorithms['torch']['available'] = False
        
        return algorithms
        
    except Exception as e:
        logger.error(f"Failed to get algorithms: {e}")
        raise HTTPException(status_code=500, detail="Failed to get algorithms")


@router.get("/methods")
async def get_available_methods():
    """Get list of available jitter analysis methods."""
    try:
        methods = {
            'jitter_dispersion': {
                'name': 'Jitter Dispersion',
                'description': 'Analyze jitter using moving IQR and averaging',
                'available': True
            },
            'ks_test': {
                'name': 'Kolmogorov-Smirnov Test',
                'description': 'Statistical test for distribution changes',
                'available': True
            }
        }
        
        return methods
        
    except Exception as e:
        logger.error(f"Failed to get methods: {e}")
        raise HTTPException(status_code=500, detail="Failed to get methods")


@router.get("/config/template")
async def get_config_template():
    """Get a configuration template."""
    try:
        config = JitterbugConfig()
        return config.dict()
        
    except Exception as e:
        logger.error(f"Failed to get config template: {e}")
        raise HTTPException(status_code=500, detail="Failed to get config template")


# Error handlers
@router.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle value errors."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponseAPI(
            error="ValidationError",
            message=str(exc),
            timestamp=datetime.now()
        ).dict()
    )


@router.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponseAPI(
            error="InternalServerError",
            message="An internal server error occurred",
            details={"exception": str(exc)},
            timestamp=datetime.now()
        ).dict()
    )