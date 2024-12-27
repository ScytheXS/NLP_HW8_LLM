"""This module contains argument bots. 
These agents should be able to handle a wide variety of topics and opponents.
They will be evaluated using methods in `evaluate.py`.
We've included a few to get your started."""

import logging
from rich.logging import RichHandler
from pathlib import Path
import random
import glob
from dialogue import Dialogue
from agents import Agent, ConstantAgent, LLMAgent
from kialo import Kialo
import tracking

# Use the same logger as agents.py, since argubots are agents;
# we split this file 
# You can change the logging level there.
log = logging.getLogger("agents")    

#############################
## Define some basic argubots
#############################

# Airhead (aka Absentia or Acephalic) always says the same thing.

airhead = ConstantAgent("Airhead", "I know right???")

# Alice is a basic prompted LLM.  You are trying to improve on Alice.
# Don't change the prompt -- instead, make a new argubot with a new prompt.

alice = LLMAgent("Alice",
                 system="You are an intelligent bot who wants to broaden your user's mind. "
                        "Ask a conversation starter question.  Then, WHATEVER "
                        "position the user initially takes, push back on it. "
                        "Try to help the user see the other side of the issue. "
                        "Answer in 1-2 sentences. Be thoughtful and polite.")

############################################################
## Other argubot classes and instances -- add your own here! 
############################################################

class KialoAgent(Agent):
    """ KialoAgent subclasses the Agent class. It responds with a relevant claim from
    a Kialo database.  No LLM is used."""
    
    def __init__(self, name: str, kialo: Kialo):
        self.name = name
        self.kialo = kialo
                
    def response(self, d: Dialogue) -> str:

        if len(d) == 0:   
            # First turn.  Just start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
        else:
            previous_turn = d[-1]['content']  # previous turn from user
            # Pick one of the top-3 most similar claims in the Kialo database,
            # restricting to the ones that list "con" arguments (counterarguments).
            neighbors = self.kialo.closest_claims(previous_turn, n=3, kind='has_cons')
            assert neighbors, "No claims to choose from; is Kialo data structure empty?"
            neighbor = random.choice(neighbors)
            log.info(f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")
            
            # Choose one of its "con" arguments as our response.
            claim = random.choice(self.kialo.cons[neighbor])
        
        return claim    
    
# Akiko doesn't use an LLM, but looks up an argument in a database.
  
akiko = KialoAgent("Akiko", Kialo(glob.glob("data/*.txt")))   # get the Kialo database from text files


###########################################
# Define your own additional argubots here!
###########################################
class ContextualKialoAgent(KialoAgent):
    def response(self, dialogue: Dialogue) -> str:
        history = [turn["content"] for turn in dialogue]
        
        weighted_query = " ".join(history[-3:])
        
        results = self.kialo.closest_claims(weighted_query, n=3, kind="has_cons")
        if results:
            reply = random.choice(results)
        else:
            reply = "I'm sorry, I couldn't find a relevant response."

        return reply 

akiki = ContextualKialoAgent("Akiki", Kialo(glob.glob("data/*.txt")))


###########################################
# Define your own additional argubots here!
###########################################
class RAGAgent(LLMAgent):
    def __init__(self, name: str, model: str, client, kialo: Kialo, **kwargs):
        super().__init__(name, model=model, client=client, **kwargs)
        self.kialo = kialo

    def response(self, d: Dialogue, **kwargs) -> str:
        # Step 1: Query Formation
        last_turn = d[-1]["content"] if d else ""
        query = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Paraphrase the following statement to make it more explicit and detailed:"},
                {"role": "user", "content": last_turn}
            ],
            model=self.model
        )
        explicit_statement = query.choices[0].message.content.strip()

        # Step 2: Retrieval
        related_claims = self.kialo.closest_claims(explicit_statement, kind="has_cons", n=3)
        claims_summary = "\n".join(related_claims)

        # Step 3: Retrieval-Augmented Generation
        final_response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"Generate a response based on the user's statement and the following related claims:\n{claims_summary}"},
                {"role": "user", "content": last_turn}
            ],
            model=self.model
        )
        return final_response.choices[0].message.content.strip()

aragorn = RAGAgent("Aragorn", model=tracking.default_model, client=tracking.default_client, kialo=Kialo(glob.glob("data/*.txt")))



###########################################
# Define your own additional argubots here!
###########################################
class AwsomAgent(LLMAgent):
    def __init__(self, name: str, model: str, client, **kwargs):
        super().__init__(name, model=model, client=client, **kwargs)

    def response(self, dialogue: Dialogue, **kwargs) -> str:
        # Step 1: Extract last human turn
        last_turn = dialogue[-1]["content"] if dialogue else ""

        # Step 2: Chain of Thought - Private Analysis
        thought_prompt = [
            {"role": "system", "content": "You are a helpful assistant that analyzes dialogue to infer intent and emotional tone."},
            {"role": "user", "content": f"Analyze the following human statement and summarize their intent, tone, and possible motivations: {last_turn}"}
        ]
        private_thought = self.client.chat.completions.create(
            messages=thought_prompt, model=self.model
        ).choices[0].message.content.strip()

        # Step 3: Generate Public Response with Private Thought Integration
        response_prompt = [
            {"role": "system", "content": "You are an assistant that responds thoughtfully to human dialogue based on analyzed context."},
            {"role": "user", "content": f"Using the following analysis: {private_thought}, generate a helpful and contextually appropriate response to the statement: {last_turn}"}
        ]
        public_response = self.client.chat.completions.create(
            messages=response_prompt, model=self.model
        ).choices[0].message.content.strip()

        return public_response


# Instantiate AwsomAgent
awsom = AwsomAgent(name="Awsom", model=tracking.default_model, client=tracking.default_client)
























