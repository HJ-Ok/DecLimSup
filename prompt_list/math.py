class prompts_math:
    def __init__(self):
        self.PREAMBLE = """As an expert problem solver solve step by step the following mathematical questions."""

        # The default gsm8k prompt from the CoT paper
        # https://arxiv.org/pdf/2201.11903.pdf page 35.

        self.PROMPT_L1 = """Question: The sum of two numbers is 6. The difference of their squares is 12. What is the positive difference of the two numbers?
Let's think step by step
Call the two numbers $x$ and $y$. 
We are given that $x+y = 6$ and $x^2 - y^2 = 12$. 
Because $x^2 - y^2$ factors into $(x+y)(x-y)$, 
we can substitute in for $x+y$, 
giving $6(x-y) = 12$, 
or $x-y = \boxed{2}$.
The answer is 2

Question: Which integer is closest to the cube root of 100?
Let's think step by step
Either 4 or 5 is closest to $\sqrt[3]{100}$, 
since $4^3=64$ and $5^3=125$.  
Since $4.5^3=91.125<100$, 
$\sqrt[3]{100}$ is closer to $\boxed{5}$ than to 4.
The answer is 5

Question: What is the value of $(x - y)(x + y)$ if $x = 10$ and $y = 15$?
Let's think step by step
$(x-y)(x+y)=(10-15)(10+15) = (-5)(25) = \boxed{-125}$.
The answer is -125

Question: If $g(x) = 3x + 7$ and $f(x) = 5x - 9$, what is the value of $f(g(8))$?
Let's think step by step
$g(8)=3(8)+7=24+7=31$. 
Thus, $f(g(8))=f(31)=5(31)-9=155-9=\boxed{146}$.
The answer is 146

Question: What is the greatest possible positive integer value of $x$ if $\displaystyle\frac{x^4}{x^2} < 10$?
Let's think step by step
On the left-hand side, $x^2$ cancels, reducing the inequality to $x^2<10$. 
Since  $3^2=9<10$ while $4^2=16>10$, the greatest possible value of $x$ is $\boxed{3}$.
The answer is 3

Question: A scale drawing of a park shows that one inch represents 800 feet. A line segment in the drawing that is 4.75 inches long represents how many feet?
Let's think step by step
Each inch of the 4.75-inch line segment represents 800 feet, 
so the whole line segment represents $4.75\times800=\frac{19}{4}\cdot800=19\cdot200=\boxed{3800}$ feet.
The answer is 3800

Question: In Mr. Abraham's class, $10$ of the $15$ students received an $A$ on the latest exam. If the same ratio of students received an $A$ on Mrs. Berkeley's latest exam, and if Mrs. Berkeley has $24$ students total, how many students in Mrs. Berkeley's class received an $A$?
Let's think step by step
If $10$ of $15$ students received an $A$, 
then the ratio of students receiving an $A$ to students not receiving an $A$ is $\frac{10}{15}$, or $\frac{2}{3}$. 
Let $x$ be the number of students in Mrs. Berkeley's class who received an $A$. 
Since the ratio is consistent across the two classes, $\frac{2}{3} = \frac{x}{24}$. 
Cross-multiplying yields $x = \frac{24\cdot 2}{3}$, so, by simplification, we can see that $\boxed{16}$ of Mrs. Berkeley's students must have received an $A$.
The answer is 16

Question: Find the value of the first term in the geometric sequence $a,b,c,32,64$.
Let's think step by step
The common ratio is $\frac{64}{32} = 2$. 
Therefore, the first term is $\frac{32}{2^3} = \frac{32}{8} = \boxed{4}$.
The answer is 4"""

        self.PROMPT_L5 = """Question: Kevin Kangaroo begins hopping on a number line at 0. He wants to get to 1, but he can hop only $\frac{1}{3}$ of the distance. Each hop tires him out so that he continues to hop $\frac{1}{3}$ of the remaining distance. How far has he hopped after five hops? Express your answer as a common fraction.
Let's think step by step
Kevin hops $1/3$ of the remaining distance with every hop. 
His first hop takes $1/3$ closer. 
For his second hop, he has $2/3$ left to travel, so he hops forward $(2/3)(1/3)$. 
For his third hop, he has $(2/3)^2$ left to travel, so he hops forward $(2/3)^2(1/3)$. 
In general, Kevin hops forward $(2/3)^{k-1}(1/3)$ on his $k$th hop. 
We want to find how far he has hopped after five hops. 
This is a finite geometric series with first term $1/3$, common ratio $2/3$, and five terms. 
Thus, Kevin has hopped $\frac{\frac{1}{3}\left(1-\left(\frac{2}{3}\right)^5\right)}{1-\frac{2}{3}} = \boxed{\frac{211}{243}}$.
The answer is \frac{211}{243}}

Question: What is the area of the region defined by the equation $x^2+y^2 - 7 = 4y-14x+3$?
Let's think step by step
We rewrite the equation as $x^2 + 14x + y^2 - 4y = 10$ and then complete the square, 
resulting in  $(x+7)^2-49 + (y-2)^2-4=10$, 
or $(x+7)^2+(y-2)^2=63$. 
This is the equation of a circle with center $(-7, 2)$ and radius $\sqrt{63},$ 
so the area of this region is $\pi r^2 = \boxed{63\pi}$.
The answer is 63\pi

Question: If $x^2+y^2=1$, what is the largest possible value of $|x|+|y|$?
Let's think step by step
If $(x,y)$ lies on the circle, 
so does $(x,-y),$ $(-x,-y),$ and $(-x,-y),$ (which all give the same value of $|x| + |y|$), 
so we can assume that $x \ge 0$ and $y \ge 0.$
Then $|x| + |y| = x + y.$  Squaring, we get
\[(x + y)^2 = x^2 + 2xy + y^2 = 1 + 2xy.\]
Note that $(x - y)^2 \ge 0.$  
Expanding, we get $x^2 - 2xy + y^2 \ge 0,$ so $2xy \le x^2 + y^2 = 1.$  
Hence,\[1 + 2xy \le 2,\]which means $x + y \le \sqrt{2}.$  
Equality occurs when $x = y = \frac{1}{\sqrt{2}},$ 
so the maximum value of $|x| + |y|$ is $\boxed{\sqrt{2}}.$
The answer is \sqrt{2}

Question: If $f(x)=\frac{ax+b}{cx+d}, abcd\not=0$ and $f(f(x))=x$ for all $x$ in the domain of $f$, what is the value of $a+d$?
Let's think step by step
The condition $f(f(x))$ means that $f$ is the inverse of itself, 
so its graph is symmetrical about the line $y = x$. 
With a rational function of this form, we will have two asymptotes: 
a vertical one at $x=-d/c$ if $cx+d$ does not divide $ax+b$, 
and a horizontal one at $y=a/c$, 
if we take the limit of $f(x)$ as $x$ goes to $\pm\infty$. 
In order for $f$ to be its own inverse, the intersection of the asymptotes must lie on the line $y=x$ 
so that it and its asymptotes reflect onto themselves. 
This means that $-d/c=a/c$, 
and therefore $-d=a$ and $a+d=\boxed{0}$.
The answer is 0

Question: A math teacher requires Noelle to do one homework assignment for each of the first five homework points she wants to earn; for each of the next five homework points, she needs to do two homework assignments; and so on, so that to earn the $n^{\text{th}}$ homework point, she has to do $n\div5$ (rounded up) homework assignments. For example, when she has 11 points, it will take $12\div5=2.4\rightarrow3$ homework assignments to earn her $12^{\text{th}}$ point. What is the smallest number of homework assignments necessary to earn a total of 25 homework points?
Let's think step by step
Noelle only has to do 1 homework assignment to earn her first point, 
and the same is true for each of her first five points.  
She must then do 2 homework assignments to earn her sixth point, seventh point, and so on, up to her tenth point. 
Continuing, we see that Noelle must do a total of \[1+1+1+1+1+2+2+2+2+2+\dots+5+5+5+5+5\] homework assignments to earn 25 points.
This sum may be rewritten as $5(1+2+3+4+5)=5(15)=\boxed{75}$.
The answer is 75

Question: The quadratic equation $x^2+mx+n=0$ has roots that are twice those of $x^2+px+m=0,$ and none of $m,$ $n,$ and $p$ is zero. What is the value of $n/p?$
Let's think step by step
Let $r_1$ and $r_2$ be the roots of $x^2+px+m=0.$ 
Since the roots of $x^2+mx+n=0$ are $2r_1$ and $2r_2,$ we have the following relationships: \[
m=r_1 r_2,\quad n=4r_1 r_2,\quad p=-(r_1+r_2), \quad\text{and}\quad
m=-2(r_1+r_2).
\] So \[
n = 4m, \quad p = \frac{1}{2}m,
\quad\text{and}\quad
\frac{n}{p}=\frac{4m}{\frac{1}{2}m}=\boxed{8}.
\]
Alternatively, the roots of \[
\left(\frac{x}{2}\right)^2 + p\left(\frac{x}{2}\right) + m = 0
\] are twice those of $x^2 + px + m = 0.$ 
Since the first equation is equivalent to $x^2 + 2px + 4m = 0,$ 
we have \[m = 2p \quad\text{and}\quad n = 4m, \quad\text{so}\quad \frac{n}{p} = \boxed{8}.\]
The answer is 8

Question: Expand $(2z^2 + 5z - 6)(3z^3 - 2z + 1)$.
Let's think step by step
$$\begin{array}{crrrrrrr}
& & & 3z^3 & & -2z & + 1 & \\
\times & & & & 2z^2 & +5z & -6 \\
\cline{1-7}\rule{0pt}{0.17in}
& & & -18z^3 & & +12z & -6 & \\
& & +15z^4 & & -10z^2 & +5z & & \\
+ & 6z^5 & & -4z^3 & +2z^2 & & & \\
\cline{1-7}\rule{0pt}{0.17in}
& 6z^5 & +15z^4 & -22z^3 & - 8z^2 &+17z & -6 &
\end{array}$$ 
The answer is 6z^5+15z^4-22z^3-8z^2+17z-6}.

Question: Find the mean of all solutions for $x$ when $x^3 + 3x^2 - 10x = 0$.
Let's think step by step
First, we factor the equation as $x(x^2 +3x - 10) = 0$.  
So, one solution is $x=0$ and the other two solutions are the solutions to $x^2 + 3x-10=0$.
We could either factor the quadratic, or note that the sum of the solutions to this quadratic is $-(3/1)=-3$, 
so the mean of the three solutions to the original equation is $-3/3=\boxed{-1}$.
The answer is -1"""

        self.PROMPT_L3 = """Question: What is the largest value of $x$ such that the expression \[\dfrac{x+1}{8x^2-65x+8}\] is not defined?
Let's think step by step
In this particular case, the fraction will be undefined only if its denominator is equal to zero. 
Because of this, we can ignore the numerator. 
We start by setting the binomial in the denominator equal to 0: 
\begin{align*} 8x^2-65x+8=0
\\\Rightarrow\qquad (8x-1)(x-8)=0
\end{align*} 
We find that the two possible values for $x$ are $\frac18$ and $8$. 
Since the question asks for the largest value, the final solution is $\boxed{8}$.
The answer is 8

Question: BoatWorks built 3 canoes in January of this year and then each subsequent calendar month they built twice the number of canoes they had built the previous month. How many total canoes were built by BoatWorks by the end of March of this year?
Let's think step by step
The number of boats built is $3+3\cdot2+3\cdot2^2 = 3+6+12 = \boxed{21}$.
The answer is 21

Question: If $5a+2b=0$ and $a$ is two less than $b$, what is $7b$?
Let's think step by step
First we begin by solving the system of equations \begin{align*}
5a+2b&=0, \\
b-2&=a.
\end{align*}
Making the substitution for $a$ from the second equation to the first, we get $5(b-2)+2b=0$, 
which simplifies to $7b-10=0$. 
Solving for $b$, we find that $b=\frac{10}{7}$. 
Hence $7b=7\cdot \frac{10}{7}=\boxed{10}$.
The answer is 10

Question: The difference between two numbers is 9, and the sum of the squares of each number is 153. What is the value of the product of the two numbers?
Let's think step by step
Call the first number $x$ and the second number $y$. 
Without loss of generality, assume $x > y$. 
We can represent the information given in the problem with the following system of linear equations:
\begin{align*}
x - y &= 9\\
x^2 + y^2 &= 153
\end{align*} 
Solving for $x$ in the first equation and substituting into the second yields $(9+y)^2 + y^2 = 153$, or $2y^2 + 18y - 72 = 0$. 
Canceling a $2$ gives $y^2 + 9y - 36 = 0$, which factors into $(y+12)(y-3)$. 
Thus, $y = 3$ and $x = 12$. So, $x \cdot y = \boxed{36}$.
The answer is 36

Question: Simplify $\frac{1}{1+\sqrt{2}}\cdot\frac{1}{1-\sqrt{2}}$.
Let's think step by step
Multiplying the numerators simply yields $1$. 
Multiplying the denominators gives $1+\sqrt{2} - \sqrt{2} -2 = 1 - 2 = -1$. 
So, the answer is $\frac{1}{-1} = \boxed{-1}$.
The answer is -1

Question: Four people can mow a lawn in 6 hours. How many more people will be needed to mow the lawn in 4 hours, assuming each person mows at the same rate?
Let's think step by step
The number of people mowing and the time required to mow are inversely proportional. 
Letting $n$ be the number of people and $t$ be the amount of time, we have $nt = (4)(6)= 24$ 
because 4 people can mow a lawn in 6 hours. 
If $m$ people can mow the lawn in 4 hours, then we must have $m(4) = 24$, so $m=6$.  
Therefore, we need $6-4 = \boxed{2}$ more people to complete the job in 4 hours.
The answer is 2

Question: BoatsRUs built 7 canoes in January of this year and then each subsequent calendar month they built twice the number of canoes they had built the previous month. How many total canoes were built by BoatsRUs by the end of May of this year?
Let's think step by step
The numbers of canoes built by BoatsRUs each month form a geometric sequence: 7, 14, 28, 56, 112. 
The first term is 7 and the common ratio is 2, so the sum of these terms is $\frac{7(2^5-1)}{2-1} = \boxed{217}$.
The answer is 217

Question: Find the coefficient of the $x^2$ term in the expansion of the product $(ax^3 + 3x^2 - 2x)(bx^2 - 7x - 4)$.
Let's think step by step
We only need to worry about the terms that multiply to have a degree of $2$. 
This would be given by the product of the terms $3x^2$ and $-4$ as well as the product of the terms $-2x$ and $-7x$. 
Thus, $$(3x^2) \times (-4) + (-2x) \times (-7x) = -12x^2 + 14x^2 = 2x^2,$$and the coefficient is $\boxed{2}$.
The answer is 2"""

        self.SHORT_PROMPT_L3 = """Question: What is the largest value of $x$ such that the expression \[\dfrac{x+1}{8x^2-65x+8}\] is not defined?
Let's think step by step
In this particular case, the fraction will be undefined only if its denominator is equal to zero. 
Because of this, we can ignore the numerator. 
We start by setting the binomial in the denominator equal to 0: 
\begin{align*} 8x^2-65x+8=0
\\\Rightarrow\qquad (8x-1)(x-8)=0
\end{align*} 
We find that the two possible values for $x$ are $\frac18$ and $8$. 
Since the question asks for the largest value, the final solution is $\boxed{8}$.
The answer is 8

Question: BoatWorks built 3 canoes in January of this year and then each subsequent calendar month they built twice the number of canoes they had built the previous month. How many total canoes were built by BoatWorks by the end of March of this year?
Let's think step by step
The number of boats built is $3+3\cdot2+3\cdot2^2 = 3+6+12 = \boxed{21}$.
The answer is 21

Question: If $5a+2b=0$ and $a$ is two less than $b$, what is $7b$?
Let's think step by step
First we begin by solving the system of equations \begin{align*}
5a+2b&=0, \\
b-2&=a.
\end{align*}
Making the substitution for $a$ from the second equation to the first, we get $5(b-2)+2b=0$, 
which simplifies to $7b-10=0$. 
Solving for $b$, we find that $b=\frac{10}{7}$. 
Hence $7b=7\cdot \frac{10}{7}=\boxed{10}$.
The answer is 10

Question: The difference between two numbers is 9, and the sum of the squares of each number is 153. What is the value of the product of the two numbers?
Let's think step by step
Call the first number $x$ and the second number $y$. 
Without loss of generality, assume $x > y$. 
We can represent the information given in the problem with the following system of linear equations:
\begin{align*}
x - y &= 9\\
x^2 + y^2 &= 153
\end{align*} 
Solving for $x$ in the first equation and substituting into the second yields $(9+y)^2 + y^2 = 153$, or $2y^2 + 18y - 72 = 0$. 
Canceling a $2$ gives $y^2 + 9y - 36 = 0$, which factors into $(y+12)(y-3)$. 
Thus, $y = 3$ and $x = 12$. So, $x \cdot y = \boxed{36}$.
The answer is 36

Question: Simplify $\frac{1}{1+\sqrt{2}}\cdot\frac{1}{1-\sqrt{2}}$.
Let's think step by step
Multiplying the numerators simply yields $1$. 
Multiplying the denominators gives $1+\sqrt{2} - \sqrt{2} -2 = 1 - 2 = -1$. 
So, the answer is $\frac{1}{-1} = \boxed{-1}$.
The answer is -1

Question: Four people can mow a lawn in 6 hours. How many more people will be needed to mow the lawn in 4 hours, assuming each person mows at the same rate?
Let's think step by step
The number of people mowing and the time required to mow are inversely proportional. 
Letting $n$ be the number of people and $t$ be the amount of time, we have $nt = (4)(6)= 24$ 
because 4 people can mow a lawn in 6 hours. 
If $m$ people can mow the lawn in 4 hours, then we must have $m(4) = 24$, so $m=6$.  
Therefore, we need $6-4 = \boxed{2}$ more people to complete the job in 4 hours.
The answer is 2

Question: BoatsRUs built 7 canoes in January of this year and then each subsequent calendar month they built twice the number of canoes they had built the previous month. How many total canoes were built by BoatsRUs by the end of May of this year?
Let's think step by step
The numbers of canoes built by BoatsRUs each month form a geometric sequence: 7, 14, 28, 56, 112. 
The first term is 7 and the common ratio is 2, so the sum of these terms is $\frac{7(2^5-1)}{2-1} = \boxed{217}$.
The answer is 217

Question: Find the coefficient of the $x^2$ term in the expansion of the product $(ax^3 + 3x^2 - 2x)(bx^2 - 7x - 4)$.
Let's think step by step
We only need to worry about the terms that multiply to have a degree of $2$. 
This would be given by the product of the terms $3x^2$ and $-4$ as well as the product of the terms $-2x$ and $-7x$. 
Thus, $$(3x^2) \times (-4) + (-2x) \times (-7x) = -12x^2 + 14x^2 = 2x^2,$$and the coefficient is $\boxed{2}$.
The answer is 2"""

        self.TEMPLATE = """
Question: """

    def get_PREAMBLE(self):
        return self.PREAMBLE

    def get_PROMPT_L1(self):
        return self.PROMPT_L1

    def get_PROMPT_L3(self):
        return self.PROMPT_L3

    def get_SHORT_PROMPT_L3(self):
        return self.SHORT_PROMPT_L3

    def get_PROMPT_L5(self):
        return self.PROMPT_L5

    def get_TEMPLATE(self):
        return self.TEMPLATE
