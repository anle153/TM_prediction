import os

import matplotlib.pyplot as plt
import numpy as np


def plotDataSeries(x=None, series=None, filename=None, title=None, path_to_dir=None, xlabel=None, ylabel=None):

    if x is not None:
        plt.plot(x, series)
    else:
        plt.plot(series)

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if (path_to_dir is not None) & (filename is not None):
        plt.savefig(path_to_dir+'/'+filename)

    plt.close()


class DataAnalysis(object):

    def __init__(self, raw_data,
                 dataset_name=GEANT_DATASET,
                 sampling_interval=5,
                 plot_dir=DATA_ANALYSIS_DIR, **kwargs):
        # Parse arguments
        self.raw_data = raw_data
        self.dataset_name = dataset_name
        self.sampling_interval = sampling_interval
        self.plot_dir = plot_dir
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

    def analyze_data_per_flow_by_day(self):
        assert self.raw_data.shape[0]

        plot_dir_by_dataset = self.plot_dir + '/' + self.dataset_name
        if not os.path.exists(plot_dir_by_dataset):
            os.makedirs(plot_dir_by_dataset)

        date_size = 24*60 / self.sampling_interval
        ndays = int(self.raw_data.shape[0] / date_size) + (self.raw_data.shape[0] % date_size > 0)

        day_means = []
        day_stds = []
        flows_means = []
        flows_stds = []
        for day in range(ndays):
            upper_bound = (day + 1) * date_size
            if upper_bound > self.raw_data.shape[0]:
                upper_bound = self.raw_data.shape[0]

            traffics_by_day = self.raw_data[day*date_size:upper_bound, :]

            flow_means = np.expand_dims(np.mean(traffics_by_day, axis=0), axis=1)
            day_means.append(np.mean(traffics_by_day))

            flow_stds = np.expand_dims(np.std(traffics_by_day, axis=0), axis=1)
            day_stds.append(np.std(traffics_by_day))

            if day==0:
                flows_means = flow_means
                flows_stds = flow_stds
            else:
                flows_means = np.concatenate((flows_means, flow_means), axis=1)
                flows_stds =np.concatenate((flows_stds, flow_stds), axis=1)

            # Plot flow_means and flow_stds
            plotDataSeries(series=flow_means,
                           filename='Flow_Mean_Day_%i.png' % (day+1),
                           title='Flow Means of Day %i' % (day+1),
                           path_to_dir=plot_dir_by_dataset,
                           xlabel='FlowID',
                           ylabel='Mbps')

            plotDataSeries(series=flow_stds,
                           filename='Flow_Std_Day_%i.png' % (day+1),
                           title='Flow Std of Day %i' % (day+1),
                           path_to_dir=plot_dir_by_dataset,
                           xlabel='FlowID',
                           ylabel='Mbps')

        # Plot day_means and days_stds
        plotDataSeries(x=range(ndays),
                       series=day_means,
                       filename='Means_by_Day.png',
                       title='Traffic Means by Date',
                       path_to_dir=plot_dir_by_dataset,
                       xlabel='Day',
                       ylabel='Mbps')

        plotDataSeries(x=range(ndays),
                       series=day_stds,
                       filename='Std_by_Day.png',
                       title='Traffic Std by Date',
                       path_to_dir=plot_dir_by_dataset,
                       xlabel='Day',
                       ylabel='Mbps')

        # Plot per flow mean over time by day
        if not os.path.exists(plot_dir_by_dataset + '/' + 'per_flow_by_day'):
            os.makedirs(plot_dir_by_dataset + '/' + 'per_flow_by_day')

        for flowID in range(flows_means.shape[0]):
            plotDataSeries(x=range(ndays),
                           series=flows_means[flowID, :],
                           filename='Mean_Flow_%i.png' % flowID,
                           title='Mean of Flow %i by Day' % flowID,
                           xlabel='Day',
                           ylabel='Mbps',
                           path_to_dir=plot_dir_by_dataset + '/' + 'per_flow_by_day')

            plotDataSeries(x=range(ndays),
                           series=flows_stds[flowID, :],
                           filename='Std_Flow_%i.png' % flowID,
                           title='Std of Flow %i by Day' % flowID,
                           xlabel='Day',
                           path_to_dir=plot_dir_by_dataset + '/' + 'per_flow_by_day')

        plot_flow_day_over_period_dir = plot_dir_by_dataset + '/' +'flow_over_period'
        if not os.path.exists(plot_flow_day_over_period_dir):
            os.makedirs(plot_flow_day_over_period_dir)
        for flowID in range(self.raw_data.shape[1]):
            filename = 'Flow_%i.png' % flowID
            plt.title('Flow %i over period' % flowID)
            plt.xlabel('Time Slot')
            plt.ylabel('Mbps')
            for day in range(ndays):
                upper_bound = (day + 1) * date_size
                if upper_bound > self.raw_data.shape[0]:
                    upper_bound = self.raw_data.shape[0]

                plt.plot(self.raw_data[day*date_size:upper_bound, flowID], label='Day % i' % day)
            if self.dataset_name == ABILENE_DATASET:
                plt.legend()
            plt.savefig(plot_flow_day_over_period_dir + '/' + filename)
            plt.close()
