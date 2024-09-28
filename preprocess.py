# Working directory; cache files and models will be stored here
work_dir = Path("./dataset")
# Directory where trained models will be saved
model_dir = Path("./drive/MyDrive/model/transformer_checkpoints")
# Last checkpoint; set to None if running for the first time. If resuming, specify the latest model.
model_checkpoint = None  # e.g., 'model_10000.pt'

# Create the working directory if it doesn't exist
if not work_dir.exists():
    work_dir.mkdir(parents=True, exist_ok=True)

# Create the model directory if it doesn't exist
if not model_dir.exists():
    model_dir.mkdir(parents=True, exist_ok=True)

# File paths for English and Chinese sentences
en_filepath = 'en_train2.txt'
cn_filepath = 'cn_train2.txt'

# Function to get the number of lines in a file
def get_line_count(filepath):
    count = 0
    with open(filepath, encoding='utf-8') as file:
        for _ in file:
            count += 1
    return count

# Number of English sentences
en_line_count = get_line_count(en_filepath)
# Number of Chinese sentences
cn_line_count = get_line_count(cn_filepath)

assert en_line_count == cn_line_count, "The number of lines in English and Chinese files do not match!"

# Total number of sentences, used for displaying progress later
total_lines = en_line_count

# Define maximum sentence length; sentences shorter than this will be padded, longer ones will be truncated
max_length = 16
print("Number of sentences:", en_line_count)
print("Maximum sentence length:", max_length)

# Initialize English and Chinese vocabularies; will be set up later
en_vocab = None
cn_vocab = None

# Define batch size; can be larger due to small memory footprint of training text
batch_size = 64
# Number of epochs; doesn't need to be too large since there are many sentences
epochs = 10
# Save the model every N steps to prevent loss due to crashes
save_after_step = 5000

# Use caching; since files are large and initialization is slow, persist initialized files
use_cache = True

# Define the training device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print("Batch size:", batch_size)
print("Saving model every {} steps".format(save_after_step))
print("Device:", device)
