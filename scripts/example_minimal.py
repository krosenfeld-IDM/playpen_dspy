"""
https://dspy-docs.vercel.app/docs/quick-start/minimal-example
"""
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)
        
if __name__ == "__main__":

    # Set up the LM
    turbo = dspy.OpenAI(model='gpt-3.5-turbo-instruct', max_tokens=250)
    # turbo = dspy.OpenAI(model='gpt-4-1106-preview', max_tokens=1000)
    dspy.settings.configure(lm=turbo)

    # Load math questions from the GSM8K dataset
    gsm8k = GSM8K()
    gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]

    print('GSM8K trainset:')
    print(gsm8k_trainset)

    # Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
    config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

    # Optimize! Use the `gsm8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
    teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
    optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset)


    # Set up the evaluator, which can be used multiple times.
    evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)

    # Evaluate our `optimized_cot` program.
    print("Evaluating...")
    evaluate(optimized_cot)

    turbo.inspect_history(n=1)

    optimized_cot(question="If I slept for 49 hours last week but each night got 1 hour less than the last night, how many hours did I sleep on the first night?")    

    turbo.inspect_history(n=1)
