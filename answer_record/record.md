
| Deepseek-math-last-layer      | GSM8K   | Note                            |
|------------------|-----------------|---------------------------------|
| Focal Loss           |     71.1       | focal_loss gamma2 alpha1 |
| CE Loss   |  69.9 |  |
| Focal Loss   |  69.9 | focal_loss gamma1 alpha1 |


| Deepseek-math-full-para       | GSM8K   | Note                            |
|------------------------------ |-----------|---------------------------------|
| Full Parameter Focal Loss     | 60.7 | {'train_runtime': 6328.4367, 'train_samples_per_second': 1.181, 'train_steps_per_second': 0.074, 'train_loss': 0.4316730402266443, 'epoch': 1.0} |
| Lora Focal Loss               |  | {'train_runtime': 1612.8406, 'train_samples_per_second': 4.633, 'train_steps_per_second': 0.29, 'train_loss': 0.5875917593978457, 'epoch': 1.0} |
