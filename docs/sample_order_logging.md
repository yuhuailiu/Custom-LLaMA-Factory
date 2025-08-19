# Sample Order Logging in LLaMAFactory

This document explains how to use the sample order logging feature in LLaMAFactory to track the order of data samples during training.

## Overview

The sample order logging feature allows you to record which samples are processed in each training batch, along with metadata about each sample. This is useful for:

- **Debugging**: Understanding which samples are being processed and in what order
- **Reproducibility**: Ensuring consistent training behavior across runs
- **Analysis**: Analyzing training patterns and sample distribution
- **Quality Control**: Verifying that data augmentation and sampling work as expected

## Features

- **Real-time logging**: Logs are written in real-time during training
- **Detailed metadata**: Records sample IDs, batch information, and timestamps
- **Multiple ID field support**: Automatically detects common ID field names (`id`, `sample_id`, `example_id`, `index`)
- **Statistics**: Provides summary statistics about the training run
- **Flexible output**: Supports both real-time logs and detailed JSON reports

## Usage

### 1. Enable in Configuration

Add the `log_sample_order` parameter to your training configuration:

```yaml
# In your YAML config file
method:
  stage: sft
  do_train: true
  finetuning_type: lora
  log_sample_order: true  # Enable sample order logging
```

### 2. Data Format Requirements

Your dataset should include an ID field. The system supports these field names:
- `id` (recommended)
- `sample_id`
- `example_id`
- `index`

Example JSONL format:
```json
{"id": "sample_001", "instruction": "Your instruction", "input": "Your input", "output": "Your output"}
{"id": "sample_002", "instruction": "Another instruction", "input": "Another input", "output": "Another output"}
```

### 3. Training Output

When `log_sample_order` is enabled, the system will generate:

- **Real-time log**: `{output_dir}/training_sample_order.log`
- **Detailed log**: `{output_dir}/detailed_sample_order.json`

## Log Formats

### Real-time Log (`training_sample_order.log`)

```
[2024-01-15 10:30:15] Batch 1: Size=2, IDs=['sample_001', 'sample_002']
[2024-01-15 10:30:16] Batch 2: Size=2, IDs=['sample_003', 'sample_004']
[2024-01-15 10:30:17] Batch 3: Size=2, IDs=['sample_005', 'sample_006']
```

### Detailed Log (`detailed_sample_order.json`)

```json
{
  "summary": {
    "total_batches": 3,
    "total_samples": 6,
    "log_file": "output/training_sample_order.log",
    "created_at": "2024-01-15 10:30:20"
  },
  "batches": [
    {
      "batch_index": 1,
      "batch_size": 2,
      "sample_ids": ["sample_001", "sample_002"],
      "metadata": [
        {
          "id": "sample_001",
          "input_length": 128,
          "label_length": 64,
          "has_images": false,
          "has_videos": false,
          "has_audios": false
        }
      ],
      "timestamp": "2024-01-15 10:30:15"
    }
  ]
}
```

## API Reference

### CustomDataCollatorWithLogging

The main class that provides sample order logging functionality.

#### Parameters

- `log_sample_order` (bool): Whether to enable logging (default: True)
- `log_file` (str): Path to the real-time log file
- `detailed_log_file` (str): Path to the detailed log file

#### Methods

- `get_sample_order_log()`: Get the in-memory log
- `save_detailed_log(filename)`: Save detailed log to file
- `clear_log()`: Clear the in-memory log
- `get_statistics()`: Get summary statistics

### LoggingPairwiseDataCollator

Specialized collator for pairwise training (RM, DPO) with logging.

### LoggingKTODataCollator

Specialized collator for KTO training with logging.

## Examples

### Basic Usage

```python
from llamafactory.data import CustomDataCollatorWithLogging

# Create custom collator
collator = CustomDataCollatorWithLogging(
    template=template,
    tokenizer=tokenizer,
    label_pad_token_id=tokenizer.pad_token_id,
    log_sample_order=True,
    log_file="logs/training_order.log"
)

# Use in training
trainer = CustomSeq2SeqTrainer(
    data_collator=collator,
    # ... other parameters
)

# After training, save detailed log
collator.save_detailed_log("logs/detailed_order.json")

# Get statistics
stats = collator.get_statistics()
print(f"Processed {stats['total_samples']} samples in {stats['total_batches']} batches")
```

### Custom Log File Path

```python
collator = CustomDataCollatorWithLogging(
    template=template,
    tokenizer=tokenizer,
    label_pad_token_id=tokenizer.pad_token_id,
    log_file="custom_path/training.log",
    detailed_log_file="custom_path/detailed.json"
)
```

## Configuration Examples

### SFT Training with Logging

```yaml
model:
  model_name_or_path: "your_model"
  trust_remote_code: true

method:
  stage: sft
  do_train: true
  finetuning_type: lora
  log_sample_order: true  # Enable logging

dataset:
  dataset: "your_dataset"
  template: "default"

output:
  output_dir: "output/with_logging"

train:
  per_device_train_batch_size: 4
  max_steps: 1000
```

### RM Training with Logging

```yaml
method:
  stage: rm
  do_train: true
  finetuning_type: lora
  log_sample_order: true

dataset:
  dataset: "your_ranking_dataset"
  template: "default"
```

## Troubleshooting

### Common Issues

1. **No ID field found**: Ensure your dataset has an `id` field
2. **Permission denied**: Check write permissions for the log directory
3. **Memory usage**: Large datasets may consume significant memory for logging

### Performance Considerations

- Logging adds minimal overhead to training
- Real-time logs are written asynchronously
- Consider disabling logging for very large datasets if memory is a concern

## Advanced Usage

### Custom Metadata Extraction

You can extend the collator to extract custom metadata:

```python
class CustomMetadataCollator(CustomDataCollatorWithLogging):
    def _log_batch_info(self, features):
        # Extract custom metadata
        for feature in features:
            if 'custom_field' in feature:
                # Process custom field
                pass
        
        # Call parent method
        super()._log_batch_info(features)
```

### Integration with Other Tools

The detailed logs can be easily integrated with:
- Data analysis tools (pandas, numpy)
- Visualization libraries (matplotlib, plotly)
- Monitoring systems (wandb, tensorboard)

## Support

For issues and questions:
1. Check the log files for error messages
2. Verify your dataset format
3. Ensure all required dependencies are installed
4. Check the LLaMAFactory documentation and issues
