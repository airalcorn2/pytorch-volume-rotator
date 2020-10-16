# PyTorch Feature Volume Rotator

![](optimize.gif)

[This function](pytorch_volume_rotator.py) uses trilinear interpolation to rotate and translate feature volumes.
Because the function is written entirely in PyTorch, it can be seamlessly plugged into a deep learning training pipeline.
As a toy example, given the following output feature volumes:

<p align="center">
  <img src="true_yaw_vol.jpg" width="250" />
  <img src="true_pitch_vol.jpg" width="250" />
  <img src="true_roll_vol.jpg" width="250" />
</p>

and the associated rotations that produced them, the true input feature volume (left) can be recovered from a randomly initialized feature volume (right):

<p align="center">
  <img src="true_in_vol.jpg" width="250">
  <img src="initial_vol.jpg" width="250">
</p>

```bash
python3 optimize_example.py
```

## Examples

```bash
python3 rotation_translation_example.py rotate
```

produces these:

<p align="center">
  <img src="yaw.gif" width="250" />
  <img src="pitch.gif" width="250" />
  <img src="roll.gif" width="250" />
</p>

```bash
python3 rotation_translation_example.py translate
```

produces these:

<p align="center">
  <img src="x.gif" width="250" />
  <img src="y.gif" width="250" />
  <img src="z.gif" width="250" />
</p>
