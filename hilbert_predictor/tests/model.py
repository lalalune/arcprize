import torch

from ..data import MAX_CONTEXT_LENGTH, NUM_TOKENS, PAD_TOKEN, MAX_SEQUENCE_LENGTH
from ..args import batch_size
from ..model import device, model


def test_model_with_zeros():
    # Create dummy input data (zeros)
    dummy_input = torch.full(
        (batch_size, MAX_SEQUENCE_LENGTH), PAD_TOKEN, dtype=torch.long
    ).to(device)
    dummy_input[:, :MAX_CONTEXT_LENGTH] = torch.randint(
        0, NUM_TOKENS, (batch_size, MAX_CONTEXT_LENGTH), device=device
    )

    # Create dummy dimensions
    dummy_dimensions = [
        [MAX_CONTEXT_LENGTH // 5, MAX_CONTEXT_LENGTH // 5]
    ]  # Example dimensions

    # Set the model to training mode
    model.train()

    # Create a dummy target (zeros)
    dummy_target = torch.full(
        (batch_size, MAX_SEQUENCE_LENGTH), PAD_TOKEN, dtype=torch.long
    ).to(device)
    dummy_target[:, :MAX_CONTEXT_LENGTH] = torch.randint(
        0, NUM_TOKENS, (batch_size, MAX_CONTEXT_LENGTH), device=device
    )

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    # Perform a forward pass
    output, _ = model(dummy_input, dummy_dimensions)

    # Calculate loss
    loss = criterion(output.view(-1, NUM_TOKENS + 1), dummy_target.view(-1))

    # Backward pass and optimization
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()

    print(f"Test completed. Loss: {loss.item()}")


def test_position_encodings():
    # Create dummy input data (zeros)
    dummy_input = torch.zeros((batch_size, MAX_SEQUENCE_LENGTH), dtype=torch.long).to(
        device
    )
    dummy_input[:, :MAX_CONTEXT_LENGTH] = torch.randint(
        0, NUM_TOKENS, (batch_size, MAX_CONTEXT_LENGTH), device=device
    )

    # Create dummy dimensions
    dummy_dimensions = [
        [MAX_CONTEXT_LENGTH // 5, MAX_CONTEXT_LENGTH // 5]
    ]  # Example dimensions

    # Perform a forward pass
    output, _ = model(dummy_input, dummy_dimensions)

    # Check if the output has the correct shape
    assert output.shape == (
        batch_size,
        MAX_SEQUENCE_LENGTH,
        NUM_TOKENS + 1,
    ), f"Expected shape {(batch_size, MAX_SEQUENCE_LENGTH, NUM_TOKENS + 1)}, but got {output.shape}"

    # Check if the position encodings are not all zeros
    assert not torch.allclose(
        output[:, :MAX_CONTEXT_LENGTH, :-1],
        torch.zeros_like(output[:, :MAX_CONTEXT_LENGTH, :-1]),
    ), "Position encodings are all zeros"

    print("Position encodings test passed.")


def count_parameters(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params


if __name__ == "__main__":
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params}")
    test_model_with_zeros()
    test_position_encodings()
