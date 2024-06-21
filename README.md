# ARC AGI 2024 Challenge
This submission is for the ARC AGI challenge. The model is a transformer model with 6 layers, 8 heads, and 512 hidden units.

The "secret sauce" here is using a Generalized Hilbert Curve function (Gilbert) to convert the 2D grid into a 1D sequence. This allows the model to learn the spatial relationships between the cells in the grid, even over distances.

To train
```
python -m hilbert_predictor.train
```

To eval
```
python -m hilbert_predictor.eval

```

## NOTES:

Tokens 0-9 are colors
Token 10 is padding, ignored by the model
Token 11 is START and token 12 is END token, which bookend the actual sequence