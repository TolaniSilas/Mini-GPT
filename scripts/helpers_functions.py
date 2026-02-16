import torch



def calculate_loss_batch(input_batch, target_batch, model, device):
    """calculates loss for a single batch."""

    # move batch to device.
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    # get model predictions.
    logits = model(input_batch)

    # calculate cross-entropy loss.
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

    return loss



def calculate_loss_loader(data_loader, model, device, num_batches=None):
    """calculates average loss across multiple batches from data loader."""

    # initialize total loss.
    total_loss = 0.0

    # handle empty data loader.
    if len(data_loader) == 0:
        return float("nan")

    # determine number of batches to process.
    if num_batches is None:
        num_batches = len(data_loader)
        
    else:
        # limit to available batches.
        num_batches = min(num_batches, len(data_loader))

    # iterate through batches and accumulate loss.
    for i, (input_batch, target_batch) in enumerate(data_loader):

        if i < num_batches:
            # calculate loss for current batch.
            loss = calculate_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()

        else:
            break

    # return average loss.
    return total_loss / num_batches



def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """evaluates model on training and validation sets."""

    # set model to evaluation mode.
    model.eval()

    # calculate losses without gradient tracking.
    with torch.no_grad():

        # calculate training loss.
        train_loss = calculate_loss_loader(train_loader, model, device, num_batches=eval_iter)

        # calculate validation loss.
        val_loss = calculate_loss_loader(val_loader, model, device, num_batches=eval_iter)

    # set model back to training mode.
    model.train()

    return train_loss, val_loss



def generate_text_simple(model, idx, max_new_tokens, context_size):
    """generates text by predicting tokens one at a time."""

    # set model to evaluation mode.
    model.eval()

    # generate tokens one by one.
    for _ in range(max_new_tokens):

        # crop context if it exceeds maximum context size.
        idx_cond = idx[:, -context_size:]

        # get model predictions without gradient tracking.
        with torch.no_grad():
            logits = model(idx_cond)

        # focus only on last time step.
        logits = logits[:, -1, :]

        # get token with highest probability (greedy decoding).
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # append sampled token to running sequence.
        idx = torch.cat((idx, idx_next), dim=1)

    return idx



def text_to_token_ids(text, tokenizer):
    """converts text to token ids with batch dimension."""

    # encode text to token ids.
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})

    # convert to tensor and add batch dimension.
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    return encoded_tensor



def token_ids_to_text(token_ids, tokenizer):
    """converts token ids back to text."""

    # remove batch dimension.
    flat = token_ids.squeeze(0)

    # decode token ids to text.
    return tokenizer.decode(flat.tolist())



def generate_and_print_sample(model, tokenizer, device, start_context):
    """generates and prints sample text from the model."""

    # set model to evaluation mode.
    model.eval()

    # get context size from positional embeddings.
    context_size = model.pos_emb.weight.shape[0]

    # encode starting context and move to device.
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    # generate text without gradient tracking.
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )

    # decode token ids back to text.
    decoded_text = token_ids_to_text(token_ids, tokenizer)

    # print generated text with newlines replaced by spaces.
    print(decoded_text.replace("\n", " "))

    # set model back to training mode.
    model.train()