"""
This is an example script to demonstrate the complete usage of the Point-to-Span estimation model.

Here, we perform the following steps:
1. Add ⧫ tokens in the `example_dataset.xml` file annotations to generate training data for the span estimation model.
2. Train the model using the generated training data.
3. Predict the spans for the `example_to_expand.xml` file using the trained model.

Basically it is just a combination of the three provided scripts with the proper arguments.

We provide some example XML files in the `example_resources` folder:
- `span_annotated.xml`: A dataset with 10 documents annotated with span annotations (start and end). Used to generate training data.
- `point_annotated.xml`: A dataset with 3 documents annotated with point annotations (⧫). Used to expand the annotations.
- `point_annotated_gold.xml`: The gold standard for the `point_annotated.xml` file. Can be used for result comparison.

All produced results are saved in the `output` folder.

@author: Gabriel Andrade
"""

import subprocess

if __name__ == "__main__":
    print("Step 1...")
    subprocess.run(["python", "generate_training_data.py",
                    "--input", "example_resources/span_annotated.xml",
                    "--output", "output/training_data",
                    "--tags", "C",
                    "--augmentation", "10",
                    "--strategy", "gaussian"])

    print("Step 2...")
    subprocess.run(["python", "train_model.py",
                    "--training_file", "output/training_data/span_annotated.xml",
                    "--output", "output/model",
                    "--tags", "C",
                    "--max_epochs", "10"])

    print("Step 3...")
    subprocess.run(["python", "estimate_span.py",
                    "--input", "example_resources/point_annotated.xml",
                    "--output", "output/example_results",
                    "--model", "output/model"])
