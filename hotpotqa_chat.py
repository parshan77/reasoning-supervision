import json
from openai import OpenAI
import random
import requests
import time

import wikienv, wrappers

client = OpenAI()

env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="dev")
env = wrappers.LoggingWrapper(env)

folder = './prompts/'
prompt_file = 'prompts_naive.json'
with open(folder + prompt_file, 'r') as f:
    prompt_dict = json.load(f)

webthink_examples = "Here are some examples." + prompt_dict['webthink_simple6']
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task."""
webthink_prompt = instruction + webthink_examples

def llm(messages, stop=["\n"]):
    response = client.chat.completions.create(
        # model="gpt-3.5-turbo-0125",
        # model="ft:gpt-3.5-turbo-0125:personal::9UOkNA6H",
        # model="ft:gpt-3.5-turbo-0125:personal::9W8xg1c5",
        # model="ft:gpt-3.5-turbo-0125:personal::9WAUgKPC",
        model="ft:gpt-3.5-turbo-0125:personal::9WAw18Yl",
        messages=messages,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop
    )
    return response.choices[0].message.content

def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1

def webthink(idx=None, prompt=webthink_prompt, to_print=True):
    question = env.reset(idx=idx)
    if to_print:
        print(idx, question)
    messages = [{"role": "system", "content": prompt}]
    messages.append({"role": "user", "content": question + "\nThought 1:"})
    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):
        n_calls += 1
        thought_action = llm(messages, stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            print('ohh...', thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            messages[-1]["content"] += f" {thought}\nAction {i}:"
            action = llm(messages, stop=[f"\n"]).strip()
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        env.record_thought(thought)
        obs = obs.replace('\\n', '')
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        messages += [
            {"role": "assistant", "content": f"{thought}\nAction {i}: {action}\nObservation {i}:"},
            {"role": "user", "content": f"{obs}\nThought {i+1}:"},
        ]
        prompt += step_str
        if to_print:
            print(step_str)
        if done:
            break
    if not done:
        obs, r, done, info = step(env, "finish[]")
    if to_print:
        print(info, '\n')
    env.close()
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return r, info

if __name__ == "__main__":
    idxs = list(range(7405))
    # idxs = list(range(90447))
    random.Random(233).shuffle(idxs)

    rs = []
    infos = []
    old_time = time.time()
    for i in idxs[:100]:
        r, info = webthink(i, to_print=False)
        rs.append(info['em'])
        infos.append(info)
        print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
        print('-----------')
        print()
