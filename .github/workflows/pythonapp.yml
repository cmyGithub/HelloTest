 {
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn import datasets\n",
    "digits=datasets.load_digits()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.],\n",
       "       [ 0.,  0., 13., 15., 10., 15.,  5.,  0.],\n",
       "       [ 0.,  3., 15.,  2.,  0., 11.,  8.,  0.],\n",
       "       [ 0.,  4., 12.,  0.,  0.,  8.,  8.,  0.],\n",
       "       [ 0.,  5.,  8.,  0.,  0.,  9.,  8.,  0.],\n",
       "       [ 0.,  4., 11.,  0.,  1., 12.,  7.,  0.],\n",
       "       [ 0.,  2., 14.,  5., 10., 12.,  0.,  0.],\n",
       "       [ 0.,  0.,  6., 13., 10.,  0.,  0.,  0.]])"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "digits.images[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x2b9184efe80>"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "plt.imshow(digits.images[3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "#将数据分成训练数据和测试数据\n",
    "from sklearn.model_selection import train_test_split\n",
    "train_x,test_x,train_y,test_y=train_test_split(digits.data,digits.target,test_size=0.3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',\n",
       "                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,\n",
       "                     weights='uniform')"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#引用knn算法，通过fit训练模型\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "model=KNeighborsClassifier()\n",
    "model.fit(train_x,train_y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9796296296296296"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#测试\n",
    "model.score(test_x,test_y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Response [200]>"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#读取一张为2的图片\n",
    "import requests\n",
    "response=requests.get(\"http://labfile.oss.aliyuncs.com/courses/777/demo.jpg\")\n",
    "response"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[  0, 144, 199, 219, 211,  41,   0,   0],\n",
       "       [  0,  74,  57,  34,  60, 255,  50,   0],\n",
       "       [  0,   0,   0,   0,   0, 158, 118,   1],\n",
       "       [  0,   1,   0,   1,   1, 254,  61,   0],\n",
       "       [  0,   0,   0,   0, 141, 255,   0,   1],\n",
       "       [  0,   0,  50, 199, 207,  18,   0,   0],\n",
       "       [  2, 254, 255, 102,   8,   7,   8,   3],\n",
       "       [ 18, 219, 239, 247, 253, 243, 245,  87]])"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from io import BytesIO\n",
    "from PIL import Image\n",
    "import numpy as np\n",
    "#Image.open(BytesIO(response.content))\n",
    "I=Image.open(BytesIO(response.content)).convert(\"L\")\n",
    "#将图片转换为数组\n",
    "demo=np.asarray(I,dtype=int)\n",
    "demo"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "plt.imshow(model,cmap=plt.cm.gray_r)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x2b919fdd978>"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAPUAAAD4CAYAAAA0L6C7AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjAsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+17YcXAAALLElEQVR4nO3d72ud9RnH8c9naaWzKoItRdqyWJViGcyWUJCisLqV+gPdg4EtTFkZiFClpQPRPds/IO7BEKWaCXZKVxVEnE5R6YSts4ndZo2OrnQ0q65NZ1E7XEl77UFOodp0uc859/09JxfvFwSTnJDvdUjf3uec3Lm/jggByOMbvR4AQL2IGkiGqIFkiBpIhqiBZOY08U0XLFgQg4ODTXzr8xw+fLjIOpJ08uTJYmtJ0uTkZLG1bBdb6+qrry621rx584qtVdKhQ4c0MTEx7Q+tkagHBwe1d+/eJr71ebZu3VpkHUnas2dPsbUkaWJiothaJf/x79y5s9ha1113XbG1ShoaGrrgbTz8BpIhaiAZogaSIWogGaIGkiFqIBmiBpIhaiAZogaSqRS17fW2P7J9wPZDTQ8FoHMzRm17QNIvJd0iaYWkjbZXND0YgM5UOVKvlnQgIg5GxClJz0m6s9mxAHSqStSLJZ37p1Djrc99he17be+1vffYsWN1zQegTVWinu7Pu867WmFEPBERQxExtHDhwu4nA9CRKlGPS1p6zsdLJB1pZhwA3aoS9buSrrV9le2LJG2Q9FKzYwHo1IwXSYiISdv3S3pN0oCkpyJif+OTAehIpSufRMQrkl5peBYANeCMMiAZogaSIWogGaIGkiFqIBmiBpIhaiCZRnboKGndunXF1rrrrruKrSVJixYtKrbWkSPlzvxdsaLcX+6OjIwUW0uSVq1aVXS96XCkBpIhaiAZogaSIWogGaIGkiFqIBmiBpIhaiAZogaSIWogmSo7dDxl+6jt90sMBKA7VY7Uv5K0vuE5ANRkxqgjYrekfxeYBUANantOzbY7QH+oLWq23QH6A69+A8kQNZBMlV9pPSvpD5KW2x63/ZPmxwLQqSp7aW0sMQiAevDwG0iGqIFkiBpIhqiBZIgaSIaogWSIGkhm1m+7c+utt/Z6hBSWLVtWbK1NmzYVW+uNN94otpYkrVy5suh60+FIDSRD1EAyRA0kQ9RAMkQNJEPUQDJEDSRD1EAyRA0kQ9RAMlWuUbbU9lu2x2zvt72lxGAAOlPl3O9JST+NiFHbl0oasf16RHzQ8GwAOlBl252PI2K09f7nksYkLW56MACdaes5te1BSSsl7ZnmNrbdAfpA5ahtXyLpeUlbI+Kzr9/OtjtAf6gUte25mgp6R0S80OxIALpR5dVvS3pS0lhEPNL8SAC6UeVIvUbS3ZLW2t7XeuNyI0CfqrLtzjuSXGAWADXgjDIgGaIGkiFqIBmiBpIhaiAZogaSIWogGaIGkpn1e2lFRLG1ps6YLafkfStpeHi42Fq7d+8utpZU/t/IdDhSA8kQNZAMUQPJEDWQDFEDyRA1kAxRA8kQNZAMUQPJVLnw4Dzbf7L959a2Oz8vMRiAzlQ5TfS/ktZGxBetSwW/Y/u3EfHHhmcD0IEqFx4MSV+0Ppzbest5UjKQQNWL+Q/Y3ifpqKTXI4Jtd4A+VSnqiDgdEddLWiJpte1vT/M1bLsD9IG2Xv2OiBOS3pa0vpFpAHStyqvfC21f3nr/m5K+J+nDpgcD0Jkqr35fKelp2wOa+p/Azoh4udmxAHSqyqvff9HUntQAZgHOKAOSIWogGaIGkiFqIBmiBpIhaiAZogaSIWogmVm/7U4/bHPSlJL3bdu2bcXWKunGG28sul4/bJXEkRpIhqiBZIgaSIaogWSIGkiGqIFkiBpIhqiBZIgaSIaogWQqR926oP97trnoINDH2jlSb5E01tQgAOpRddudJZJuk7S92XEAdKvqkfpRSQ9KOnOhL2AvLaA/VNmh43ZJRyNi5P99HXtpAf2hypF6jaQ7bB+S9JyktbafaXQqAB2bMeqIeDgilkTEoKQNkt6MiB81PhmAjvB7aiCZti5nFBFva2orWwB9iiM1kAxRA8kQNZAMUQPJEDWQDFEDyRA1kMys33anpNHR0aLrbd68udhaX375ZbG1jh8/Xmyt0vphGyiO1EAyRA0kQ9RAMkQNJEPUQDJEDSRD1EAyRA0kQ9RAMkQNJFPpNNHWlUQ/l3Ra0mREDDU5FIDOtXPu93cjYqKxSQDUgoffQDJVow5Jv7M9Yvve6b6AbXeA/lA16jURsUrSLZI2277p61/AtjtAf6gUdUQcaf33qKQXJa1ucigAnauyQd5825eefV/SOknvNz0YgM5UefV7kaQXW1d0mCPp1xHxaqNTAejYjFFHxEFJ3ykwC4Aa8CstIBmiBpIhaiAZogaSIWogGaIGkiFqIJnGtt05c+ZMU9/6KwYGBoqs0wuPP/54sbXuueeeYmuV3Jqm5HZCkjR37tyi602HIzWQDFEDyRA1kAxRA8kQNZAMUQPJEDWQDFEDyRA1kAxRA8lUitr25bZ32f7Q9pjtG5oeDEBnqp77/QtJr0bED21fJOniBmcC0IUZo7Z9maSbJP1YkiLilKRTzY4FoFNVHn4vk3RM0rDt92xvb13/+yvYdgfoD1WiniNplaTHImKlpJOSHvr6F7HtDtAfqkQ9Lmk8Iva0Pt6lqcgB9KEZo46ITyQdtr289ambJX3Q6FQAOlb11e8HJO1ovfJ9UNKm5kYC0I1KUUfEPklDDc8CoAacUQYkQ9RAMkQNJEPUQDJEDSRD1EAyRA0kQ9RAMo3spTU5OakTJ0408a3Pc8011xRZR5ImJiaKrSVJW7ZsKbbWfffdV2yt+fPP+yO/xpw+fbrYWpI0PDxcZJ1PP/30grdxpAaSIWogGaIGkiFqIBmiBpIhaiAZogaSIWogGaIGkpkxatvLbe875+0z21tLDAegfTOeJhoRH0m6XpJsD0j6p6QXG54LQIfaffh9s6S/R8Q/mhgGQPfajXqDpGenu+HcbXeOHz/e/WQAOlI56tY1v++Q9Jvpbj93250rrriirvkAtKmdI/UtkkYj4l9NDQOge+1EvVEXeOgNoH9Uitr2xZK+L+mFZscB0K2q2+78RxJPlIFZgDPKgGSIGkiGqIFkiBpIhqiBZIgaSIaogWSIGkjGEVH/N7WPSWr3zzMXSCq7r005We8b96t3vhURC6e7oZGoO2F7b0QM9XqOJmS9b9yv/sTDbyAZogaS6aeon+j1AA3Ket+4X32ob55TA6hHPx2pAdSAqIFk+iJq2+ttf2T7gO2Hej1PHWwvtf2W7THb+21v6fVMdbI9YPs92y/3epY62b7c9i7bH7Z+djf0eqZ29fw5dWuDgL9p6nJJ45LelbQxIj7o6WBdsn2lpCsjYtT2pZJGJP1gtt+vs2xvkzQk6bKIuL3X89TF9tOSfh8R21tX0L04Ik70eq529MORerWkAxFxMCJOSXpO0p09nqlrEfFxRIy23v9c0pikxb2dqh62l0i6TdL2Xs9SJ9uXSbpJ0pOSFBGnZlvQUn9EvVjS4XM+HleSf/xn2R6UtFLSnt5OUptHJT0o6UyvB6nZMknHJA23nlpstz2/10O1qx+i9jSfS/N7NtuXSHpe0taI+KzX83TL9u2SjkbESK9nacAcSaskPRYRKyWdlDTrXuPph6jHJS095+Mlko70aJZa2Z6rqaB3RESWyyuvkXSH7UOaeqq01vYzvR2pNuOSxiPi7COqXZqKfFbph6jflXSt7ataL0xskPRSj2fqmm1r6rnZWEQ80ut56hIRD0fEkogY1NTP6s2I+FGPx6pFRHwi6bDt5a1P3Sxp1r2wWem6302KiEnb90t6TdKApKciYn+Px6rDGkl3S/qr7X2tz/0sIl7p4UyY2QOSdrQOMAclberxPG3r+a+0ANSrHx5+A6gRUQPJEDWQDFEDyRA1kAxRA8kQNZDM/wD5eMFeed67wwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(demo,cmap=plt.cm.gray_r)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2])"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#模型的预测\n",
    "model.predict(np.atleast_2d(demo.flatten()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    ""
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3.0
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
