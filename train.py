criteria = TranslationLoss()
writer = SummaryWriter(log_dir='runs/transformer_loss')
torch.cuda.empty_cache()

# Initialize the step counter
step = 0

if model_checkpoint:
    # Extract the step number from the model checkpoint filename
    step = int(model_checkpoint.replace("model_", "").replace(".pt", ""))

model.train()
for epoch in range(epochs):
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for index, data in loop:
        # Generate data
        src, tgt, tgt_y, n_tokens = data
        src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)

        # Clear gradients
        optimizer.zero_grad()
        # Perform transformer computation
        out = model(src, tgt)
        # Pass the result through the final linear layer for prediction
        out = model.predictor(out)

        """
        Calculate the loss. Since during training we predict all outputs, we need to reshape 'out'.
        Our 'out' has the shape (batch_size, sequence_length, vocabulary_size), after reshaping it becomes:
        (batch_size * sequence_length, vocabulary_size).
        Among these prediction results, we only need the non-<pad> parts, so we need to normalize, that is,
        divide by n_tokens.
        """
        loss = criteria(
            out.contiguous().view(-1, out.size(-1)),
            tgt_y.contiguous().view(-1)
        ) / n_tokens
        # Compute gradients
        loss.backward()
        # Update parameters
        optimizer.step()

        loop.set_description(f"Epoch {epoch + 1}/{epochs}")
        loop.set_postfix(loss=loss.item())
        loop.update(1)

        step += 1

        del src
        del tgt
        del tgt_y

        if step != 0 and step % save_after_step == 0:
            torch.save(model, model_dir / f"model_{step}.pt")
