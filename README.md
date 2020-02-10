# Tramission
 A tramission model for 2019-nCov

## Model
 Classify the people into four categories by isolated or not and sicked or not:
 
 Number | sicked | healthy
 ---- | ---- | ---- 
 Isolated | I | K
 Not isolated | J | ∞
 
 We have:
 1. dJ = (b - a) * J * dt
 2. dI = (a * J - r * I) * dt
 
 where a denotes the isolation rate, b the infection rate and r the cure rate.
 
 By solving the partial differential equation above, we get:
 I = I0 * exp(-r * t) + (a * J0 / (b - a + r)) * exp((b - a) * t)
 
 Furthur modification:
 the isolation rate a = sigmoid(pt + b) and increases with the time.
