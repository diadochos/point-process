from scipy.stats import expon, uniform


class PointProcessGenerator:
    @classmethod
    def inhomo_simulation(kls, measure, delta_t, measure_sup: float, T):
        t=0
        points_homo=[]
        points_inhomo=[]
        while t < T:
            points_homo.append(t)
            r = expon.rvs(scale=1/measure_sup) #scale=1/lambda
            t+=r
            if t >= T:
                break
            D = uniform.rvs(loc=0,scale=1)
            if D * measure_sup <= measure[int(t/delta_t)]:
                points_inhomo.append(t)
        if points_inhomo[-1] > T:
            del points_inhomo[-1]
        del points_homo[0]

        return points_inhomo
