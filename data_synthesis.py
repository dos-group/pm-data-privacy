import sys

from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator


def suppress_output(function):

    # Prevent printing to stdout during function execution
    def silent_function(*args, **kwargs):
        original_stdout = sys.stdout
        null = open('/dev/null', 'w')  # On Unix-like systems
        # null = open('nul', 'w')  # On Windows
        sys.stdout = null
        result = function(*args, **kwargs)
        null.close()
        sys.stdout = original_stdout
        return result
    return silent_function


@suppress_output
def create_synthetic_data(df_original, num_synthetic_samples):

    dataset_file = '/tmp/original_data.csv'
    df_original.to_csv(dataset_file, index=False)

    description_file = f'/tmp/description.json'
    describer = DataDescriber(category_threshold=0)
    candidate_keys = {key: False for key in df_original.keys()}

    describer.describe_dataset_in_correlated_attribute_mode(
        dataset_file=dataset_file,
        epsilon=0, # Severity of added noise to distributions
        k=3, # The max number of parents in Bayesian network, i.e., max number of incoming edges.
        seed=42,
        attribute_to_is_candidate_key=candidate_keys
    )
    describer.save_dataset_description_to_file(description_file)

    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_synthetic_samples, description_file)
    df_synthetic = generator.synthetic_dataset
    return df_synthetic

