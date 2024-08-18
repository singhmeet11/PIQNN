# PIQNN(Physics Inspired Quantum Neural Network)

Physics-Informed Neural Networks (PINNs) are neural networks that incorporate physical laws, represented by differential equations, into the learning process. By embedding these constraints, PINNs reduce the space for possible solutions for a particular physical system. 

Here we embed parameterized quantum circuits which work as quantum layers into the PINN model, there by creating novel PIQNN. These hybrid quantum classical models are then trained for 2 cases, first we test the working on a
first order ODE and then we solve the heat equation in 2D with particular boundary conditions and initial condition. 



## Heat equation solution

The heat equation is 

$$\frac{\partial u}{\partial t} = \alpha(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} )$$

we work with the boundary conditions given by u(x, y=1, t) = 100; u(x=1, y, t) = 0; u(x, y=0, t) = 0; u(x=0, y0, t) = 0 and see the evolution of the heat map in the 2D space. 
![image](https://github.com/user-attachments/assets/ede3a40c-10e9-4896-b12a-edada494446d)
For impirically testing which ansatz to choose we selected 5 ansatz from Sim et. al's paper and evaluated if there was any correlation between expressibility and trainig accuracy of the model. 

Results show that the best quantum model shows a smaller loss value of 1.06%. However we do not find any correlation between  expressibility and trainig accuracy. 

![image](https://github.com/user-attachments/assets/45eff971-704a-4b03-868f-c1b9b97ebbaf)


