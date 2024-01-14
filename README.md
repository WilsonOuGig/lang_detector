# llm_demo



credit to https://www.youtube.com/watch?v=kCc8FmEb1nY



### Notes help to understanding the architecture:


batch=16
block=24 # context length


#### 1. get random batches

x:  [16, 24] # 24 token ids
y: 	[16, 24] # 24 token ids, shift-right by one



#### 2. embedding x (embedding size=64):  

tok_emb = [16, 24, 64]
position_emb = [24, 64]

x_embeding = embedding +  position (broadcast to 16 batches)   
x_embeding: [16, 24, 64]


#### 3. self-attension (head size=16)

k : [16, 24, 16]
q : [16, 24, 16]
wei = q * k'T  [16, 24, 24]  -- lower triangular part of the matrix    

v : [16, 24, 16]

v = wei*v   [16, 24, 16] -- each token is  aggregation of the preceding tokens and itself based on wei


output of self-attention [16, 24, 16]



#### 4. output of self-attention multi-head

head_size = n_embd // n_head
so stack all heads output, the multi-head output is:

out:  [16, 24, 64]


#### 5.  feed forward :

input:

multi-head output + residual  :   [16, 24, 64]

(layer normalization put before multi-head input)

output:

feed forward output + residual  :   [16, 24, 64]

(layer normalization put before feed-forward input)


6 repeat blocks (2,3,5) for many times.

#### 7.  linear

input: [16, 24, 64]
output:  [16, 24, 50257] # vocab size

#### 8. soft-max:
  logits:  [384, 50257]
  target(y): [384, ]

  cross_entropy -> loss



