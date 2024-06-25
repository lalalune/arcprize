# ARC AGI 2024 Challenge
This submission is for the ARC AGI challenge.

The "hilbert_predictor" incorporates several novel features and is under ongoing research.

If you're interested in collaborating or are working on ARC AGI and just want people to hang out with while you do it, we have a Discord here: https://discord.gg/SQy6vA5eE4

To train
```
python -m hilbert_predictor.train
```

To eval
```
python -m hilbert_predictor.eval

```

To train with wandb
```
wandb login
python -m hilbert_predictor.train --wandb
```

## NOTES:

Tokens 0-9 are colors
Token 10 is padding, ignored by the model
Token 11 is START and token 12 is END token, which bookend the actual sequence
