class prompts_strategyqa:
    def __init__(self):
        self.PREAMBLE = """"""

        # The default gsm8k prompt from the CoT paper
        # https://arxiv.org/pdf/2201.11903.pdf page 35.

        self.PROMPT = """Q: Do hamsters provide food for any animals?
A: Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. So the answer is yes.

Q: Could Brooke Shields succeed at University of Pennsylvania?
A: Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. So the answer is yes.

Q: Yes or no: Hydrogen’s atomic number squared exceeds number of Spice Girls?
A: Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5. So the answer is no.

Q: Yes or no: Is it common to see frost during some college commencements?
A: College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. So the answer is yes.

Q: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?
A: The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. So the answer is no.

Q: Yes or no: Would a pear sink in water?
A: The density of a pear is about 0.6g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float. So the answer is no"""

        self.TEMPLATE = """
Q: {question}
A:"""

    def get_PREAMBLE(self):
        return self.PREAMBLE

    def get_PROMPT(self):
        return self.PROMPT

    def get_SHORT_PROMPT(self):
        return self.SHORT_PROMPT

    def get_SIMPLE_PROMPT(self):
        return self.SIMPLE_PROMPT

    def get_SHORT_SIMPLE_PROMPT(self):
        return self.SHORT_SIMPLE_PROMPT

    def get_EXTRA_3_SHOTS(self):
        return self.EXTRA_3_SHOTS

    def get_TEMPLATE(self):
        return self.TEMPLATE
