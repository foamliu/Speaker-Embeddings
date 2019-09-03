import matplotlib.pyplot as plt

if __name__ == '__main__':
    k = 0.2
    warmup_steps = 4000
    init_lr = 512 ** (-0.5)

    lr_list = []
    for step_num in range(1, 50000):
        # print(step_num)
        lr = k * init_lr * min(step_num ** (-0.5), step_num * (warmup_steps ** (-1.5)))
        # print(lr_1)
        # print(lr_2)
        lr_list.append(lr)

        # if step_num > 20:
        #     break

    print(lr_list[:100])
    print(lr_list[-100:])

    plt.plot(lr_list)
    plt.show()
