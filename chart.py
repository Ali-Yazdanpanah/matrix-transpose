import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()


# data to plot
n_groups = 7
means_int_prallel = (0.0173, 0.0174, 0.0337, 0.0925, 0.3320, 1.1361, 7.4527)
means_double_parallel = (0.0063 ,0.0103 ,0.0364 ,0.1331, 0.5147, 9.5517, 9.5517)
means_int_compare = (0.0056/0.0173, 2.1499/0.0174, 8.5973/0.0337, 49.7879/0.0925, 631.6230/0.3320, 2850.8076/1.1361, 11060.1019/7.4527)
means_double_compare = (0.0037/0.0063 ,2.0188/0.0103 ,9.5363/0.0364 ,77.0618/0.1331, 603.2632/0.5147, 2901.5592/9.5517, 11719.8570/9.5517)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_int_compare, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Int Comparison')

rects2 = plt.bar(index + bar_width, means_double_compare, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Double Comparison')

objects = ('32*32', '512*512', '1024*1024', '2048*2048',
           '4096*4096', '8192*8192', '16384*16384')

plt.ylabel('Speed Up')
plt.xlabel('Matrix size')
plt.title('Time elapsed')
plt.xticks(index + bar_width/2, objects)
plt.legend()
plt.yscale("linear")
plt.tight_layout()
plt.show()
