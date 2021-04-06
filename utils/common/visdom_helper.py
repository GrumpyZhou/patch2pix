import visdom
import numpy as np

class VisMeter:
    def __init__(self, name, vis=None, env='', ptit=''):
        self.name = name
        self.meter = []
        self.vis = vis
        self.env = env
        self.ptit = ptit
        self.opts = dict(mode='lines', showlegend=True, 
                        layoutopts={'plotly': dict(title=ptit, 
                                                   xaxis={'title': 'iters'})})    
    def clear(self):
        self.meter = []
        
    def append(self, x):
        self.meter.append(x)
        
    def mean(self):
        if len(self.meter) > 0:
            return np.mean(self.meter)
        else:
            return None
        
    def plot(self, epoch):
        if self.vis is None:
            return
        
        if self.mean() is None:
            return
            
        X = [epoch]
        Y = [self.mean()]
        self.vis.line(X=X, Y=Y, env=self.env, 
                      win=self.ptit, 
                      name=self.name, 
                      opts=self.opts, 
                      update='append')
        self.vis.save(envs=[self.env])

    def __repr__(self):
        return 'Visdom meter(env={}, plot={}, name={})'.format(self.env, self.ptit, self.name)


class VisPlots:
    def __init__(self, plots, vis, env, prefix='train'):
        """
        plots: namespace(plot1=namespace(legend1, legend2), plot2=namespace(legend1, legend2)..) 
        """
        self.vis = vis
        self.env = env
        
        for name in plots.__dict__:
            self.init_plot_meters('{}.{}'.format(prefix, name), plots.__dict__[name])
        self.plots = plots

    def init_plot_meters(self, name, plot):
        """
        name: plot name
        plot: Namespace(leg1=None, leg2=None)
        """
        for legend in plot.__dict__ :
            plot.__dict__[legend] = VisMeter(legend, self.vis, self.env, ptit=name)

    def plot(self, epoch):
        plots = self.plots        
        for name in plots.__dict__: 
            plot = plots.__dict__[name]
            for legend in plot.__dict__:
                plot.__dict__[legend].plot(epoch)    
    
    def clear(self):
        plots = self.plots
        for name in plots.__dict__: 
            plot = plots.__dict__[name]
            for legend in plot.__dict__:
                plot.__dict__[legend].clear()

    def get_plot_print(self, plot):
        """
        plot: Namespace(leg1=None, leg2=None)
        """
        mprint = ''
        for legend in plot.__dict__:
            meter = plot.__dict__[legend]
            val = meter.mean()
            if val:
                mprint = '{}{}={:.2f} '.format(mprint, legend, meter.mean())
        return mprint
    
    
    
    
    
