class prompts_arc_c:
    def __init__(self):

        self.PROMPT = """Follow the given examples and answer the question.
        
Q: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat? 
(A) dry palms (B) wet palms (C) palms covered with oil (D) palms covered with lotion
A: Let's think step by step. 
Friction produces heat. Dry skin has a higher coefficient of friction than wet or oily skin. Therefore, dry palms will produce the most heat when rubbed together.
Therefore, the answer is (A).

Q: Which of the following statements best explains why magnets usually stick to a refrigerator door? 
(A) The refrigerator door is smooth. (B) The refrigerator door contains iron. (C) The refrigerator door is a good conductor. (D) The refrigerator door has electric wires in it.
A: Let's think step by step. 
Magnets are attracted to ferrous materials, which contain iron. The refrigerator door is usually made of steel, which is an alloy of iron and carbon. The presence of iron in the refrigerator door allows the magnet to stick to it.
Therefore, the answer is (B).

Q: A fold observed in layers of sedimentary rock most likely resulted from the 
(A) cooling of flowing magma. (B) converging of crustal plates. (C) deposition of river sediments. (D) solution of carbonate minerals.
A: Let's think step by step. 
Sedimentary rocks are formed from the accumulation and compaction of sediments, such as sand, mud, and organic matter. 
When these sediments are deposited, they are typically laid down in horizontal layers. 
If these layers are later subjected to compressional forces, such as those that occur when two crustal plates converge, they can be folded. 
The folding of the layers is a result of the deformation of the rock under the influence of stress. 
The other options (A, C, and D) are less likely to cause folding in sedimentary rocks. 
Cooling of flowing magma (A) can cause the formation of igneous rocks, but it does not typically cause folding in sedimentary rocks. 
Deposition of river sediments (C) can result in the formation of sedimentary layers, but it does not typically cause folding. 
Solution of carbonate minerals (D) can cause the formation of caves and other karst features, but it does not typically cause folding in sedimentary rocks.
Therefore, The answer is (B).

Q: Which of these do scientists offer as the most recent explanation as to why many plants and animals died out at the end of the Mesozoic era? 
(A) worldwide disease (B) global mountain building (C) rise of mammals that preyed upon plants and animals (D) impact of an asteroid created dust that blocked the sunlight
A: Let's think step by step. 
The question asks for the most recent explanation for the mass extinction at the end of the Mesozoic era. 
Answer choice (A), worldwide disease, is not a widely accepted explanation for the mass extinction. 
Answer choice (B), global mountain building, is not a widely accepted explanation for the mass extinction. 
Answer choice (C), rise of mammals that preyed upon plants and animals, is not a widely accepted explanation for the mass extinction. 
Answer choice (D), impact of an asteroid created dust that blocked the sunlight, is the most widely accepted explanation for the mass extinction. 
Therefore, the answer is (D).

Q: A boat is acted on by a river current flowing north and by wind blowing on its sails. The boat travels northeast. In which direction is the wind most likely applying force to the sails of the boat? 
(A) west (B) east (C) north (D) south
A: Let's think step by step. 
The boat is traveling northeast, which means it is moving both north and east.
The river current is flowing north, so it is not causing the boat to move east. 
Therefore, the wind must be blowing on the sails of the boat in a direction that causes the boat to move east. 
The only option that satisfies this requirement is (B) east.
Therefore, the answer is (B)."""

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
