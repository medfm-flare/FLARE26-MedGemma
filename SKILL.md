---
name: mle
description: Customize the MLE codebase
---

## Goal

Based on the user's request (usually like "Train / fine-tune a certain model on a certain dataset"), fill in the
template codebase by implementing the abstract functions in the engine.

## Constraints

1. You MUST NOT modify any files other than "mle/engine/preprocess.py", "mle/engine/train.py", and
   "mle/engine/evaluate.py".
2. You MUST follow the signature as well as the docstrings of the abstract functions in the engine files, in addition to
   the user and skill instructions.

## Steps

1. Read the user's request and identify what model and dataset to use. If the user does not specify any of the two,
   prompt the user to specify.
2. Verify the dataset exists at the location specified by the user. If the user does not specify, prompt the user to
   decide whether to download the dataset. Note that this path is only a local path, which SHOULD NEVER appear in the
   codebase.
3. Read this template codebase and understand what the codebase is for.
4. Read the instructions of the desired dataset format. If the user does not specify, search online to determine the
   most suitable format based on the context.
5. Modify "mle/engine/preprocess.py" to implement the `preprocess` function that converts the dataset to the desired
   format (if different) and applies the preprocessing transformations and analysis if applicable.
6. Read the instructions of how to train or fine-tune the model. If the user does not specify, search online for the
   official documentation of the model. If none can be found, infer the best setup based on other similar models.
7. Modify "mle/engine/train.py" to implement the `train` function that trains the model.
8. Read the instructions of how to evaluate the model. If the user does not specify, search online for the official
   documentation of the dataset. If none can be found, infer the best evaluation protocol based on the dataset content.
9. Modify "mle/engine/evaluate.py" to implement the `evaluate` function that evaluates the model.