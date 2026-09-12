# Change Point Detection Algorithm Selection Guide

This guide helps users select the most appropriate change point detection algorithm for network measurement time series analysis. All algorithms are specifically designed and tuned for RTT (Round-Trip Time) measurements and network congestion detection.

## 📊 Available Algorithms

### 1. Ruptures (Default - Recommended)
**Algorithm**: `ruptures`

**Description**: State-of-the-art change point detection using the ruptures library with multiple kernel functions.

**Strengths**:
- Fast and accurate
- Multiple kernel options (RBF, L1, L2, Normal)
- Well-tested and maintained
- Good performance on various data types
- Configurable penalty parameters

**Best for**:
- Network monitoring in production environments
- Large RTT datasets (>1000 measurements)
- Real-time congestion detection
- When you need reliable, fast results for network analysis

**Configuration**:
```yaml
change_point_detection:
  algorithm: "ruptures"
  threshold: 0.25
  ruptures_model: "rbf"    # Options: "rbf", "l1", "l2", "normal"
  ruptures_penalty: 10.0   # Higher = fewer change points
```

**CLI Usage**:
```bash
jitterbug analyze data.csv --algorithm ruptures --threshold 0.25
```

### 2. Bayesian Change Point (BCP)
**Algorithm**: `bcp`

**Description**: Classical Bayesian approach using Student-t likelihood and constant priors.

**Strengths**:
- Probabilistic framework
- Good theoretical foundation
- Handles uncertainty well
- Works well with small datasets

**Weaknesses**:
- Slower than ruptures
- Requires external dependency
- Less configurable

**Best for**:
- Network research and analysis
- When you need probabilistic confidence in congestion detection
- Small to medium RTT datasets (<500 measurements)
- Academic studies requiring theoretical interpretability

**Configuration**:
```yaml
change_point_detection:
  algorithm: "bcp"
  threshold: 0.25
  min_time_elapsed: 1800
```

**CLI Usage**:
```bash
jitterbug analyze data.csv --algorithm bcp --threshold 0.25
```

### 3. PyTorch Neural Network
**Algorithm**: `torch`

**Description**: Deep learning approach using CNN+LSTM architecture for pattern recognition.

**Strengths**:
- Can learn complex patterns
- Potentially higher accuracy with training
- Good for non-linear change patterns
- Handles noisy data well

**Weaknesses**:
- Requires PyTorch installation
- More computationally intensive
- May need training for optimal performance
- Less interpretable

**Best for**:
- Complex network congestion patterns
- RTT datasets with non-linear behavior
- When you have GPU resources available
- Research into ML-based network analysis

**Configuration**:
```yaml
change_point_detection:
  algorithm: "torch"
  threshold: 0.25
  min_time_elapsed: 1800
```

**CLI Usage**:
```bash
jitterbug analyze data.csv --algorithm torch --threshold 0.25
```

## 🎯 Algorithm Selection Matrix

| Criteria | Ruptures | BCP | PyTorch |
|----------|----------|-----|---------|
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Accuracy** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Interpretability** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Configurability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Memory Usage** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Dependencies** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

## 🔍 Detailed Algorithm Comparison

### Performance Characteristics

| Algorithm | Time Complexity | Space Complexity | Min RTT Points |
|-----------|----------------|------------------|----------------|
| Ruptures | O(n log n) | O(n) | 10 |
| BCP | O(n²) | O(n) | 5 |
| PyTorch | O(n) | O(n) | 50 |

### Parameter Sensitivity

#### Ruptures Parameters
- **ruptures_model**:
  - `"rbf"`: Radial Basis Function (default, good for most cases)
  - `"l1"`: L1 norm (good for sparse changes)
  - `"l2"`: L2 norm (good for gradual changes)
  - `"normal"`: Normal distribution (good for Gaussian data)
- **ruptures_penalty**: Controls number of change points (1-100, default: 10)

#### BCP Parameters
- **threshold**: Detection sensitivity (0.01-0.5, default: 0.25)
- **min_time_elapsed**: Minimum time between change points (seconds)

#### PyTorch Parameters
- **threshold**: Detection sensitivity (0.1-0.5, default: 0.25)
- Uses internal neural network parameters (automatically configured)

## 📈 Use Case Recommendations

### Network Monitoring (Production)
**Recommended**: Ruptures with RBF kernel
```yaml
change_point_detection:
  algorithm: "ruptures"
  ruptures_model: "rbf"
  ruptures_penalty: 10.0
  threshold: 0.25
```

### Research & Analysis
**Recommended**: BCP for interpretability, Ruptures for speed
```yaml
change_point_detection:
  algorithm: "bcp"  # or "ruptures"
  threshold: 0.2    # More sensitive for research
```

### Real-time Applications
**Recommended**: Ruptures with L2 kernel
```yaml
change_point_detection:
  algorithm: "ruptures"
  ruptures_model: "l2"
  ruptures_penalty: 15.0  # Fewer false positives
```

### Noisy Data
**Recommended**: PyTorch or Ruptures with higher penalty
```yaml
change_point_detection:
  algorithm: "torch"  # or "ruptures"
  ruptures_penalty: 20.0  # For ruptures
  threshold: 0.3      # Less sensitive
```

## 🛠️ Configuration Examples

### High Sensitivity (Detect More Change Points)
```yaml
change_point_detection:
  algorithm: "ruptures"
  threshold: 0.15
  ruptures_penalty: 5.0
  min_time_elapsed: 900  # 15 minutes
```

### Low Sensitivity (Detect Fewer Change Points)
```yaml
change_point_detection:
  algorithm: "ruptures"
  threshold: 0.35
  ruptures_penalty: 20.0
  min_time_elapsed: 3600  # 1 hour
```

### Balanced Configuration (Default)
```yaml
change_point_detection:
  algorithm: "ruptures"
  threshold: 0.25
  ruptures_penalty: 10.0
  min_time_elapsed: 1800  # 30 minutes
```

## 🎮 Interactive Selection

### Quick Selection Questions

1. **What's your primary goal?**
   - Fast, reliable detection → **Ruptures**
   - Research/interpretability → **BCP**
   - Complex pattern detection → **PyTorch**

2. **What's your RTT dataset size?**
   - Small (<100 measurements) → **BCP**
   - Medium (100-1000 measurements) → **Ruptures**
   - Large (>1000 measurements) → **Ruptures** or **PyTorch**

3. **What's your tolerance for false positives?**
   - Low (prefer fewer detections) → Higher penalty/threshold
   - High (prefer more detections) → Lower penalty/threshold

4. **What are your computational resources?**
   - Limited → **Ruptures** or **BCP**
   - Abundant → **PyTorch**

### Decision Tree

```
RTT Dataset Size?
├── Small (<100 measurements) → BCP
├── Medium (100-1000 measurements)
│   ├── Need interpretability? → BCP
│   └── Need speed? → Ruptures
└── Large (>1000 measurements)
    ├── Have GPU? → PyTorch
    ├── Complex network patterns? → PyTorch
    └── Production monitoring? → Ruptures
```

## 💡 Best Practices

### General Guidelines
1. **Start with Ruptures**: It's fast, accurate, and well-tested
2. **Use BCP for research**: When you need theoretical grounding
3. **Try PyTorch for complex data**: When traditional methods fail
4. **Adjust sensitivity**: Start with defaults, then fine-tune

### Performance Optimization
1. **Large datasets**: Use Ruptures with higher penalty
2. **Real-time**: Use Ruptures with L2 kernel
3. **Batch processing**: Any algorithm works well
4. **Memory constraints**: Use BCP or Ruptures

### Troubleshooting
1. **Too many change points**: Increase penalty/threshold
2. **Too few change points**: Decrease penalty/threshold
3. **Inconsistent results**: Check data quality and preprocessing
4. **Slow performance**: Switch to Ruptures or increase min_time_elapsed

## 📊 Example Configurations for Network Scenarios

### Production Network Monitoring
```yaml
change_point_detection:
  algorithm: "ruptures"
  ruptures_model: "rbf"
  ruptures_penalty: 12.0
  threshold: 0.25
  min_time_elapsed: 1800  # 30 minutes
```

### High-Frequency RTT Analysis
```yaml
change_point_detection:
  algorithm: "ruptures"
  ruptures_model: "l2"
  ruptures_penalty: 8.0
  threshold: 0.2
  min_time_elapsed: 300  # 5 minutes
```

### Network Research Studies
```yaml
change_point_detection:
  algorithm: "bcp"
  threshold: 0.15
  min_time_elapsed: 600  # 10 minutes
```

### Complex Network Pattern Detection
```yaml
change_point_detection:
  algorithm: "torch"
  threshold: 0.3
  min_time_elapsed: 1200  # 20 minutes
```

## 🔬 Advanced Topics

### Custom Algorithm Development
For advanced users who want to implement custom algorithms:

1. Inherit from `BaseChangePointDetector`
2. Implement the `detect()` method
3. Return list of `ChangePoint` objects
4. Register in `ChangePointDetector._create_algorithm()`

### Algorithm Evaluation
Use the built-in benchmarking to compare algorithms:

```python
from jitterbug.evaluation import AlgorithmBenchmark

benchmark = AlgorithmBenchmark(dataset)
results = benchmark.compare_algorithms(['ruptures', 'bcp', 'torch'])
```

This guide should help you select the most appropriate algorithm for your specific use case and data characteristics.
