https://huggingface.co/datasets/Hypersniper/riddles_v1?clone=true
---
license: apache-2.0
language:
- en
---
# Riddle Processing with GPT-4

[Buy me Ko-fi](https://ko-fi.com/hypersniper)

## Credits
All credit for the original riddles goes to [crawsome's GitHub repository](https://github.com/crawsome/riddles).

## Project Overview
This project involves processing each riddle using GPT-4. The correct answers were provided to the model to generate a desirable output focused on reasoning and logical breakdown.

- `riddles.json` (`riddles_1`) — 386 samples, sourced from [crawsome's GitHub repository](https://github.com/crawsome/riddles).
- `riddles_2.json` — 83 samples, sourced from various Google searches.

### Prompt Structure
The following prompt was used to guide GPT-4 responses:

> "Answer the user's question using the answer. Do not mention you have the answer to the riddle. You are to answer it as if you never got the answer. Prioritize educational insights by breaking down the logic behind your decisions. Make sure to cover all relevant topics in a logical order."

### Goal
The primary goal of this dataset was to fine-tune a language model in outputting structured and logical reasoning. 

## Performance
The performance of this small dataset is noteworthy. It demonstrates a proficient ability in breaking down a question logically and providing clear, reasoned answers.

## Example Outputs

### Fine-Tuning Details
- **Model:** Open Orca Mistral 7B
- **Fine-Tuning:** One epoch with the specified dataset

### Sample Questions and Outputs

#### Riddle 1
**Question:** _"I am easy to lift, but hard to throw. What am I?"_

**Output:** 
The riddle describes an object that is lightweight and easy to lift, yet not suitable for throwing due to its properties, like lacking aerodynamics or a solid shape. A feather fits this description as it is lightweight, easy to lift, but not designed for throwing, being used instead for insulation, flight, and decoration.

**Correct Answer:** Feather

#### Riddle 2
**Question:** _"Rachel goes to the supermarket and buys 10 tomatoes. Unfortunately, on the way back home, all but 9 get ruined. How many tomatoes are left in a good condition?"_

**Output:**
Breaking down the problem: Rachel buys 10 tomatoes, but only 1 gets ruined, as 'all but 9' implies. Therefore, 9 tomatoes remain in good condition.

**Correct Answer:** 9

