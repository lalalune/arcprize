import torch
from .model import (
    DecoderOnlyTransformer,
    NUM_TOKENS,
    d_model,
    nhead,
    num_layers,
    dim_feedforward,
    max_seq_length,
    dropout_rate,
    device,
    batch_size,
)


def test_model_with_zeros():
    # Create a model instance
    model = DecoderOnlyTransformer(
        NUM_TOKENS,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        max_seq_length,
        dropout_rate,
        device,
    )

    # Create dummy input data (zeros)
    dummy_input = torch.zeros((batch_size, max_seq_length), dtype=torch.long).to(device)

    # Set the model to training mode
    model.train()

    # Create a dummy target (zeros)
    dummy_target = torch.zeros((batch_size, max_seq_length), dtype=torch.long).to(
        device
    )

    # Create an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Perform a forward pass
    output = model(dummy_input)

    # Calculate loss
    loss = criterion(output.view(-1, NUM_TOKENS + 1), dummy_target.view(-1))

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Test completed. Loss: {loss.item()}")


if __name__ == "__main__":
    test_model_with_zeros()
