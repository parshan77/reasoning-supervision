import json

trajs_files = [
    "trajs/0125.json",
    "trajs/0125_1.json",
    "trajs/0125_2.json"
]
all_file = "0125_all_ex_fix.jsonl"
correct_file = "0125_correct_ex_fix.jsonl"

folder = './prompts/'
prompt_file = 'prompts_naive.json'
with open(folder + prompt_file, 'r') as f:
    prompt_dict = json.load(f)

webthink_examples = "\nHere are some examples." + prompt_dict['webthink_simple6']
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task."""
instruction += webthink_examples

if __name__ == "__main__":
    all_f = open(all_file, "w")
    correct_f = open(correct_file, "w")
    
    counter_all = 0
    counter_correct = 0
    for trajs_file in trajs_files:
        with open(trajs_file, "r") as f:
            trajs = json.load(f)
        for traj in trajs:
            question = traj["observations"][0]
            observations = traj["observations"][1:]
            thoughts = traj["thoughts"]
            actions = traj["actions"]
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": question + "\nThought 1:"}
            ]
            for i, (obs, thought, action) in enumerate(zip(observations, thoughts, actions)):
                action = action[0].upper() + action[1:]
                messages.append({"role": "assistant", "content": f"{thought}\nAction {i+1}: {action}\nObservation {i+2}:"})
                messages.append({"role": "user", "content": f"{obs}\nThought {i+2}:"})
            messages = messages[:-1]

            all_f.write(json.dumps({"messages": messages}) + "\n")
            counter_all += 1
            if traj["em"]:
                correct_f.write(json.dumps({"messages": messages}) + "\n")
                counter_correct += 1
    all_f.close()
    correct_f.close()
    print("{}/{} trajectories lead to correct answer.".format(counter_correct, counter_all))
