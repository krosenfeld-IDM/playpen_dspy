"""
https://dspy-docs.vercel.app/docs/building-blocks/signatures
"""
import dspy

if __name__ == "__main__":

    # Set up the LM
    turbo = dspy.OpenAI(model='gpt-3.5-turbo-instruct', max_tokens=250)
    dspy.settings.configure(lm=turbo)

    # Example A: Sentiment classification
    sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.
    classify = dspy.Predict('sentence -> sentiment')
    print(sentence)
    print("sentiment:", classify(sentence=sentence).sentiment)

    # Example B: Summary
    print('')
    # Example from the XSum dataset.
    document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."""

    summarize = dspy.ChainOfThought('document -> summary')
    response = summarize(document=document)
    
    print("Summary:", response.summary)    
    print("Rationale:", response.rationale)


    # Example C: Classification
    print('')
    class Emotion(dspy.Signature):
        """Classify emotion among sadness, joy, love, anger, fear, surprise."""
        
        sentence = dspy.InputField()
        sentiment = dspy.OutputField()

    sentence = "i started feeling a little vulnerable when the giant spotlight started blinding me"  # from dair-ai/emotion

    classify = dspy.Predict(Emotion)
    print("Classification:", classify(sentence=sentence).sentiment)  

    # Example D: Faithfulness
    print('')    

    class CheckCitationFaithfulness(dspy.Signature):
        """Verify that the text is based on the provided context."""

        context = dspy.InputField(desc="facts here are assumed to be true")
        text = dspy.InputField()
        faithfulness = dspy.OutputField(desc="True/False indicating if text is faithful to context")

    context = "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."

    text = "Lee scored 3 goals for Colchester United."

    faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
    print(faithfulness(context=context, text=text))