
2025.11.19

previous branch: master-colldata-dist_dxdydq  merged from master-colldata-dist_dxdydq-iscoll_head-- 

1. main model (dev_gru): 
    architecture: RNN --> repr vector
    input: 5 context depth image, 5 context odom poses 
    output: two heads, future waypoints + the pose (dx, dy, qw, qz) of SG (could be new SG) wrt baselink
    
2. coll pred model:
    input: curr depth image
    output: collision logit
    
3. model weights:
    devgru-two_heads-mixed_25K, devgru-two_heads-mixed_45K 

