#!/usr/bin/env python
"""
Interactive algorithm selection guide for Jitterbug.

This interactive script helps users select the most appropriate
change point detection algorithm based on their specific needs.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jitterbug.models import ChangePointDetectionConfig, JitterAnalysisConfig, JitterbugConfig


class AlgorithmSelector:
    """Interactive algorithm selection assistant."""
    
    def __init__(self):
        """Initialize the algorithm selector."""
        self.user_preferences = {}
        self.recommendations = []
    
    def ask_question(self, question: str, options: List[str], default: Optional[str] = None) -> str:
        """Ask a multiple choice question."""
        print(f"\n❓ {question}")
        for i, option in enumerate(options, 1):
            marker = " (default)" if option == default else ""
            print(f"   {i}. {option}{marker}")
        
        while True:
            try:
                choice = input(f"\nEnter your choice (1-{len(options)}): ").strip()
                if not choice and default:
                    return default
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(options):
                    return options[choice_idx]
                else:
                    print(f"Please enter a number between 1 and {len(options)}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\n\nSelection cancelled.")
                sys.exit(0)
    
    def ask_yes_no(self, question: str, default: bool = None) -> bool:
        """Ask a yes/no question."""
        default_text = " (Y/n)" if default is True else " (y/N)" if default is False else " (y/n)"
        print(f"\n❓ {question}{default_text}")
        
        while True:
            try:
                answer = input("Enter your answer: ").strip().lower()
                if not answer and default is not None:
                    return default
                
                if answer in ['y', 'yes', 'true', '1']:
                    return True
                elif answer in ['n', 'no', 'false', '0']:
                    return False
                else:
                    print("Please enter 'y' for yes or 'n' for no")
            except KeyboardInterrupt:
                print("\n\nSelection cancelled.")
                sys.exit(0)
    
    def ask_number(self, question: str, min_val: float = None, max_val: float = None, default: float = None) -> float:
        """Ask for a numeric input."""
        range_text = ""
        if min_val is not None and max_val is not None:
            range_text = f" ({min_val}-{max_val})"
        elif min_val is not None:
            range_text = f" (min: {min_val})"
        elif max_val is not None:
            range_text = f" (max: {max_val})"
        
        default_text = f" (default: {default})" if default is not None else ""
        
        print(f"\n❓ {question}{range_text}{default_text}")
        
        while True:
            try:
                answer = input("Enter your answer: ").strip()
                if not answer and default is not None:
                    return default
                
                value = float(answer)
                
                if min_val is not None and value < min_val:
                    print(f"Value must be at least {min_val}")
                    continue
                
                if max_val is not None and value > max_val:
                    print(f"Value must be at most {max_val}")
                    continue
                
                return value
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\n\nSelection cancelled.")
                sys.exit(0)
    
    def gather_user_preferences(self):
        """Gather user preferences through interactive questions."""
        print("🎯 Welcome to the Jitterbug Algorithm Selection Guide!")
        print("=" * 60)
        print("I'll ask you a few questions to help select the best algorithm for your needs.")
        
        # Primary use case
        use_case = self.ask_question(
            "What is your primary use case for network measurement analysis?",
            [
                "Real-time network monitoring",
                "Batch analysis of historical RTT data",
                "Network research and experimentation",
                "Production congestion alerting",
                "Network performance analysis"
            ],
            default="Real-time network monitoring"
        )
        self.user_preferences['use_case'] = use_case
        
        # Data characteristics
        data_size = self.ask_question(
            "How much RTT data do you typically analyze?",
            [
                "Small datasets (< 100 RTT measurements)",
                "Medium datasets (100-1000 RTT measurements)",
                "Large datasets (1000-10000 RTT measurements)",
                "Very large datasets (> 10000 RTT measurements)"
            ],
            default="Medium datasets (100-1000 RTT measurements)"
        )
        self.user_preferences['data_size'] = data_size
        
        # Performance requirements
        performance_priority = self.ask_question(
            "What is your performance priority?",
            [
                "Speed is critical (real-time processing)",
                "Accuracy is most important",
                "Balanced speed and accuracy",
                "Memory efficiency is key",
                "Interpretability is essential"
            ],
            default="Balanced speed and accuracy"
        )
        self.user_preferences['performance_priority'] = performance_priority
        
        # Computational resources
        has_gpu = self.ask_yes_no(
            "Do you have GPU resources available?",
            default=False
        )
        self.user_preferences['has_gpu'] = has_gpu
        
        # Noise tolerance
        noise_level = self.ask_question(
            "How noisy are your RTT measurements typically?",
            [
                "Low noise (clean network measurements)",
                "Moderate noise (some network variation)",
                "High noise (significant network jitter)",
                "Very high noise (unstable network conditions)"
            ],
            default="Moderate noise (some network variation)"
        )
        self.user_preferences['noise_level'] = noise_level
        
        # False positive tolerance
        false_positive_tolerance = self.ask_question(
            "What is your tolerance for false congestion alerts?",
            [
                "Very low (prefer to miss some congestion)",
                "Low (some false alerts okay)",
                "Moderate (balanced approach)",
                "High (better to catch all congestion)"
            ],
            default="Moderate (balanced approach)"
        )
        self.user_preferences['false_positive_tolerance'] = false_positive_tolerance
        
        # Experience level
        experience_level = self.ask_question(
            "What is your experience level with network analysis?",
            [
                "Beginner (just getting started with network analysis)",
                "Intermediate (some network monitoring experience)",
                "Advanced (experienced network analyst)",
                "Expert (deep understanding of network performance)"
            ],
            default="Intermediate (some network monitoring experience)"
        )
        self.user_preferences['experience_level'] = experience_level
        
        # Special requirements
        needs_interpretability = self.ask_yes_no(
            "Do you need highly interpretable results (e.g., for network research)?",
            default=False
        )
        self.user_preferences['needs_interpretability'] = needs_interpretability
        
        # Real-time requirements
        if "real-time" in use_case.lower() or "production" in use_case.lower():
            max_latency = self.ask_number(
                "What is your maximum acceptable latency (seconds)?",
                min_val=0.1,
                max_val=60.0,
                default=5.0
            )
            self.user_preferences['max_latency'] = max_latency
    
    def analyze_preferences(self):
        """Analyze user preferences and generate recommendations."""
        print("\n🔍 Analyzing your preferences...")
        
        # Algorithm scoring
        algorithm_scores = {
            'ruptures': 0,
            'bcp': 0,
            'torch': 0
        }
        
        reasons = {
            'ruptures': [],
            'bcp': [],
            'torch': []
        }
        
        # Use case scoring
        use_case = self.user_preferences['use_case']
        if "real-time" in use_case.lower():
            algorithm_scores['ruptures'] += 3
            reasons['ruptures'].append("Excellent for real-time processing")
            algorithm_scores['bcp'] += 1
            algorithm_scores['torch'] += 1
        elif "research" in use_case.lower():
            algorithm_scores['bcp'] += 3
            reasons['bcp'].append("Highly interpretable for research")
            algorithm_scores['ruptures'] += 2
            algorithm_scores['torch'] += 2
        elif "production" in use_case.lower():
            algorithm_scores['ruptures'] += 3
            reasons['ruptures'].append("Reliable for production systems")
            algorithm_scores['bcp'] += 2
            algorithm_scores['torch'] += 1
        
        # Data size scoring
        data_size = self.user_preferences['data_size']
        if "Small" in data_size:
            algorithm_scores['bcp'] += 2
            reasons['bcp'].append("Good for small datasets")
            algorithm_scores['ruptures'] += 1
        elif "Medium" in data_size:
            algorithm_scores['ruptures'] += 3
            reasons['ruptures'].append("Optimal for medium datasets")
            algorithm_scores['bcp'] += 2
            algorithm_scores['torch'] += 1
        elif "Large" in data_size or "Very large" in data_size:
            algorithm_scores['ruptures'] += 3
            reasons['ruptures'].append("Scales well with large datasets")
            algorithm_scores['torch'] += 2
            reasons['torch'].append("Handles large datasets efficiently")
            algorithm_scores['bcp'] += 1
        
        # Performance priority scoring
        performance = self.user_preferences['performance_priority']
        if "Speed is critical" in performance:
            algorithm_scores['ruptures'] += 3
            reasons['ruptures'].append("Fastest algorithm")
            algorithm_scores['bcp'] += 1
            algorithm_scores['torch'] += 1
        elif "Accuracy is most important" in performance:
            algorithm_scores['bcp'] += 3
            reasons['bcp'].append("High accuracy with uncertainty quantification")
            algorithm_scores['torch'] += 2
            reasons['torch'].append("Can achieve high accuracy with proper tuning")
            algorithm_scores['ruptures'] += 2
        elif "Memory efficiency" in performance:
            algorithm_scores['bcp'] += 3
            reasons['bcp'].append("Memory efficient")
            algorithm_scores['ruptures'] += 2
            algorithm_scores['torch'] += 1
        elif "Interpretability" in performance:
            algorithm_scores['bcp'] += 3
            reasons['bcp'].append("Highly interpretable results")
            algorithm_scores['ruptures'] += 2
            algorithm_scores['torch'] += 1
        
        # GPU availability
        if self.user_preferences['has_gpu']:
            algorithm_scores['torch'] += 2
            reasons['torch'].append("Can leverage GPU acceleration")
        
        # Noise level scoring
        noise_level = self.user_preferences['noise_level']
        if "Low noise" in noise_level:
            algorithm_scores['ruptures'] += 2
            algorithm_scores['bcp'] += 2
            algorithm_scores['torch'] += 1
        elif "High noise" in noise_level or "Very high noise" in noise_level:
            algorithm_scores['torch'] += 3
            reasons['torch'].append("Robust to high noise levels")
            algorithm_scores['ruptures'] += 1
            algorithm_scores['bcp'] += 1
        
        # False positive tolerance
        fp_tolerance = self.user_preferences['false_positive_tolerance']
        if "Very low" in fp_tolerance:
            algorithm_scores['bcp'] += 2
            reasons['bcp'].append("Conservative detection approach")
            algorithm_scores['ruptures'] += 1
        elif "High" in fp_tolerance:
            algorithm_scores['torch'] += 2
            algorithm_scores['ruptures'] += 2
        
        # Experience level
        experience = self.user_preferences['experience_level']
        if "Beginner" in experience:
            algorithm_scores['ruptures'] += 2
            reasons['ruptures'].append("Easy to use with good defaults")
        elif "Expert" in experience:
            algorithm_scores['torch'] += 2
            reasons['torch'].append("Highly configurable for experts")
            algorithm_scores['bcp'] += 2
        
        # Interpretability needs
        if self.user_preferences['needs_interpretability']:
            algorithm_scores['bcp'] += 3
            reasons['bcp'].append("Provides probabilistic confidence measures")
            algorithm_scores['ruptures'] += 1
            algorithm_scores['torch'] += 0
        
        # Real-time latency requirements
        if 'max_latency' in self.user_preferences:
            max_latency = self.user_preferences['max_latency']
            if max_latency < 1.0:
                algorithm_scores['ruptures'] += 3
                reasons['ruptures'].append("Meets strict latency requirements")
            elif max_latency < 5.0:
                algorithm_scores['ruptures'] += 2
                algorithm_scores['bcp'] += 1
        
        # Store results
        self.recommendations = []
        for algo, score in algorithm_scores.items():
            self.recommendations.append({
                'algorithm': algo,
                'score': score,
                'reasons': reasons[algo]
            })
        
        # Sort by score
        self.recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    def generate_configuration(self, algorithm: str) -> Dict[str, Any]:
        """Generate recommended configuration for the selected algorithm."""
        config = {
            'algorithm': algorithm,
            'threshold': 0.25,
            'min_time_elapsed': 1800
        }
        
        # Adjust based on user preferences
        fp_tolerance = self.user_preferences['false_positive_tolerance']
        if "Very low" in fp_tolerance:
            config['threshold'] = 0.35
        elif "High" in fp_tolerance:
            config['threshold'] = 0.15
        
        # Algorithm-specific adjustments
        if algorithm == 'ruptures':
            performance = self.user_preferences['performance_priority']
            if "Speed is critical" in performance:
                config['ruptures_model'] = 'l2'
                config['ruptures_penalty'] = 15.0
            elif "Accuracy is most important" in performance:
                config['ruptures_model'] = 'rbf'
                config['ruptures_penalty'] = 8.0
            else:
                config['ruptures_model'] = 'rbf'
                config['ruptures_penalty'] = 10.0
        
        # Real-time adjustments
        if 'max_latency' in self.user_preferences:
            if self.user_preferences['max_latency'] < 1.0:
                config['min_time_elapsed'] = 900  # 15 minutes
        
        return config
    
    def present_recommendations(self):
        """Present recommendations to the user."""
        print("\n🎯 Algorithm Recommendations")
        print("=" * 60)
        
        if not self.recommendations:
            print("❌ No recommendations could be generated.")
            return
        
        # Show top 3 recommendations
        for i, rec in enumerate(self.recommendations[:3], 1):
            algorithm = rec['algorithm']
            score = rec['score']
            reasons = rec['reasons']
            
            print(f"\n#{i}. {algorithm.upper()}")
            print(f"   Score: {score}/15")
            
            if reasons:
                print("   Reasons:")
                for reason in reasons:
                    print(f"     • {reason}")
            
            # Generate configuration
            config = self.generate_configuration(algorithm)
            print(f"   Recommended config: {config}")
        
        # Ask user to select
        print("\n" + "=" * 60)
        selected_algorithm = self.ask_question(
            "Which algorithm would you like to use?",
            [rec['algorithm'] for rec in self.recommendations[:3]] + ["Show all options"],
            default=self.recommendations[0]['algorithm']
        )
        
        if selected_algorithm == "Show all options":
            print("\n📋 All Algorithm Options:")
            for rec in self.recommendations:
                print(f"   {rec['algorithm']}: {rec['score']}/15")
            
            selected_algorithm = self.ask_question(
                "Select your algorithm:",
                ['ruptures', 'bcp', 'torch'],
                default=self.recommendations[0]['algorithm']
            )
        
        return selected_algorithm
    
    def generate_final_config(self, selected_algorithm: str):
        """Generate final configuration file."""
        print(f"\n⚙️  Generating configuration for {selected_algorithm}...")
        
        # Base configuration
        base_config = self.generate_configuration(selected_algorithm)
        
        # Ask for jitter analysis method
        jitter_method = self.ask_question(
            "Which jitter analysis method would you prefer?",
            ["jitter_dispersion", "ks_test"],
            default="jitter_dispersion"
        )
        
        # Ask for output format
        output_format = self.ask_question(
            "What output format would you prefer?",
            ["json", "csv", "parquet"],
            default="json"
        )
        
        # Create full configuration
        full_config = {
            'change_point_detection': base_config,
            'jitter_analysis': {
                'method': jitter_method,
                'threshold': 0.25,
                'moving_average_order': 6,
                'moving_iqr_order': 4
            },
            'latency_jump': {
                'threshold': 0.5
            },
            'data_processing': {
                'minimum_interval_minutes': 15,
                'outlier_detection': True
            },
            'output_format': output_format,
            'verbose': False
        }
        
        # Adjust based on preferences
        if self.user_preferences['experience_level'] == "Beginner (just getting started)":
            full_config['verbose'] = True
        
        return full_config
    
    def save_configuration(self, config: Dict[str, Any]):
        """Save configuration to file."""
        # Save as YAML
        try:
            import yaml
            config_file = Path(__file__).parent / "recommended_config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"✅ Configuration saved to: {config_file}")
        except ImportError:
            # Save as JSON if YAML not available
            config_file = Path(__file__).parent / "recommended_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✅ Configuration saved to: {config_file}")
        
        return config_file
    
    def provide_usage_examples(self, config: Dict[str, Any], config_file: Path):
        """Provide usage examples with the generated configuration."""
        print("\n💡 Usage Examples")
        print("=" * 40)
        
        algorithm = config['change_point_detection']['algorithm']
        
        print("📋 Command Line Usage:")
        print(f"   jitterbug analyze data.csv --config {config_file.name}")
        print(f"   jitterbug analyze data.csv --algorithm {algorithm}")
        
        print("\n🐍 Python API Usage:")
        print(f"""   from jitterbug import JitterbugAnalyzer, JitterbugConfig
   
   # Load configuration
   config = JitterbugConfig.from_file('{config_file.name}')
   
   # Create analyzer
   analyzer = JitterbugAnalyzer(config)
   
   # Analyze data
   results = analyzer.analyze_from_file('data.csv')
   
   # Get congestion periods
   congested_periods = results.get_congested_periods()
   print(f"Found {{len(congested_periods)}} congestion periods")""")
        
        print("\n🔧 Configuration Customization:")
        print("   You can edit the configuration file to fine-tune parameters:")
        print(f"   - Threshold: {config['change_point_detection']['threshold']}")
        print(f"   - Algorithm: {algorithm}")
        print(f"   - Jitter method: {config['jitter_analysis']['method']}")
        
        # Algorithm-specific tips
        if algorithm == 'ruptures':
            print("\n💡 Ruptures-specific tips:")
            print("   - Increase penalty for fewer change points")
            print("   - Try different models (rbf, l1, l2) for different data types")
            print("   - Use 'l2' model for faster processing")
        elif algorithm == 'bcp':
            print("\n💡 BCP-specific tips:")
            print("   - Lower threshold for more sensitive detection")
            print("   - Results include confidence probabilities")
            print("   - Good for small datasets and research")
        elif algorithm == 'torch':
            print("\n💡 PyTorch-specific tips:")
            print("   - Requires PyTorch installation")
            print("   - May benefit from GPU acceleration")
            print("   - Good for complex, non-linear patterns")


def main():
    """Run the interactive algorithm selector."""
    try:
        selector = AlgorithmSelector()
        
        # Gather user preferences
        selector.gather_user_preferences()
        
        # Analyze preferences
        selector.analyze_preferences()
        
        # Present recommendations
        selected_algorithm = selector.present_recommendations()
        
        # Generate final configuration
        final_config = selector.generate_final_config(selected_algorithm)
        
        # Save configuration
        config_file = selector.save_configuration(final_config)
        
        # Provide usage examples
        selector.provide_usage_examples(final_config, config_file)
        
        print("\n🎉 Algorithm selection complete!")
        print("   You're ready to start analyzing your data with Jitterbug!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Selection cancelled. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please try again or refer to the documentation.")


if __name__ == "__main__":
    main()