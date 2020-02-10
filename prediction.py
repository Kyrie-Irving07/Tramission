import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


def read_data(path):
    data = []
    date = []
    real_date = []
    file = open(path, 'r')
    for line in file:
        time = int(line.split(' ')[0])
        number = int(line.split(' ')[1])
        data.append(number)
        if date:
            real_date.append(time - date[0])
        else:
            real_date.append(0)
        date.append(time)
    real_date = np.reshape(real_date, [-1, 1])
    data = np.reshape(data, [-1, 1])
    return data, real_date


def plot_i(i, time, name='I', path='./visual/'):
    plt.close('all')
    plt.plot(time, i, "b.-", label='Diagnosis')
    plt.title("Tramission Prediction")
    plt.xlabel("Time/Day")
    plt.ylabel("Number of Patients")
    plt.grid(linestyle='-.')
    # plt.legend()

    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + "/"+name+".png")


def plot_i_and_j(i, j, time, name='I_J', path='./visual/'):
    plt.close('all')
    plt.plot(time, i, "b.-", label='I')
    plt.plot(time, j, "r.-", label='J')
    plt.title("Tramission Prediction")
    plt.xlabel("Time/Day")
    plt.ylabel("Number of Patients")
    plt.legend()

    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + "/"+name+".png")


def plot_i_and_label(i, label, time, name='I_Label', path='./visual/'):
    plt.close('all')
    plt.plot(time, i, "b.-", label='I')
    plt.plot(time, label, "r.-", label='Label')
    plt.title("Tramission Prediction")
    plt.xlabel("Time/Day")
    plt.ylabel("Number of Patients")
    plt.legend()

    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + "/"+name+".png")


def plot_energy(energy, step, name='energy', path='./energy/'):
    plt.close('all')
    plt.plot(step, energy, "b.-", label='Energy')
    plt.title("Energy Information")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()

    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + "/energy_"+name+".png")


def func_alpha(name, time, alpha0, beta, inflection=18, t0=0):
    # Alpha = Sigmoid(p*t + b)
    # Alpha(t0) = alpha0, Alpha(t0 + inflection) = beta.
    arc_alpha0 = np.log(alpha0 / (1 - alpha0))
    arc_beta = np.log(beta / (1 - beta))
    p_init = (arc_alpha0 - arc_beta) / (t0 - inflection)
    b_init = arc_alpha0 - t0 * p_init
    p = tf.get_variable(name + '_p', dtype=tf.float32, initializer=float(p_init))
    b = tf.get_variable(name + '_b', dtype=tf.float32, initializer=float(b_init))
    alpha = tf.sigmoid(p * time + b, name)
    return alpha


def parameter(name, init):
    hidden_init = np.log(init / (1 - init))
    hidden = tf.get_variable(name + '_hidden', dtype=tf.float32, initializer=float(hidden_init))
    param = tf.sigmoid(hidden, name=name)
    return param


class Model:
    def __init__(self, alpha=0.8, beta=0.7, gamma=0.9, j0=0, i0=0):
        self.time = tf.placeholder(tf.float32, [None, 1], 'time')
        # self.alpha = tf.get_variable('alpha', dtype=tf.float32, initializer=alpha)
        # self.beta = tf.get_variable('beta', dtype=tf.float32, initializer=beta)
        # self.gamma = tf.get_variable('gamma', dtype=tf.float32, initializer=gamma)
        self.alpha = func_alpha('alpha', self.time, alpha, beta)
        self.beta = parameter('beta', beta)
        self.gamma = parameter('gamma', gamma)
        self.j0 = tf.nn.relu(tf.get_variable('j0_raw', dtype=tf.float32, initializer=float(i0 * 2.)), 'j0')
        # self.j0 = j0
        self.i0 = i0
        self.i = self.predict_i(self.time)
        self.i_label = tf.placeholder(tf.float32, [None, 1], 'label_I')
        self.j = self.predict_j(self.time)
        self.energy = tf.nn.l2_loss((self.i - self.i_label))

        self.optimizer = tf.train.AdamOptimizer(5e-3).minimize(self.energy)
        self.save_folder = './model/'
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.saver = tf.train.Saver(max_to_keep=2)

    def predict_i(self, t):
        return self.i0 * tf.exp(-self.gamma * t) + \
               self.alpha * self.j0 * tf.exp((self.beta - self.alpha) * t) / (self.beta - self.alpha + self.gamma)

    def predict_j(self, t):
        return self.j0 * tf.exp((self.beta - self.alpha) * t)

    def train(self, data, date, epoch=20000, c=False):
        sess = tf.Session()
        latest = tf.train.latest_checkpoint(self.save_folder)
        if latest and c:
            self.saver.restore(sess, latest)
        else:
            sess.run(tf.global_variables_initializer())

        total_energy = []
        step = []
        temp_energy = []
        for i in range(epoch + 1):
            _, energy = sess.run([self.optimizer, self.energy], feed_dict={self.time: date, self.i_label: data})
            temp_energy.append(energy)
            if i % 20 == 0 and i != 0:
                print('Epoch %d| Energy %05f' % (i, np.average(temp_energy)))
                total_energy.append(np.average(temp_energy))
                temp_energy = []
                step.append(i)
                plot_energy(total_energy, step)
            if i % 100 == 0 and i != 0:
                self.saver.save(sess, self.save_folder, i)

    def predict(self, date):
        sess = tf.Session()
        latest = tf.train.latest_checkpoint(self.save_folder)
        self.saver.restore(sess, latest)
        i, j = sess.run([self.i, self.j], feed_dict={self.time: date})
        return i, j


data, date = read_data('data.txt')
print(data)

mymodel = Model(i0=data[0])
mymodel.train(data, date, 20000, c=True)

time = np.reshape(range(2 * len(date)), [-1, 1])
i, j = mymodel.predict(time)
plot_i_and_j(i, j, time)

plot_i(i, time)
print(i)

i, j = mymodel.predict(date)
plot_i_and_label(i, data, date)


