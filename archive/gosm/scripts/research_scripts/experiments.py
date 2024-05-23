"""
This script calls the main functions of evaluation_cui.py and
evaluate_percentage.py and writes the results in a .tex file. These .tex
files provide afterwards the right format for the table environment in latex.
So the file can be used for a direct import from Latex.
"""

import argparse
import evaluate_percentage
import evaluate_cui
import percentage

parser = argparse.ArgumentParser()

parser.add_argument('--data-file',
                    help='The name of the file containing power generation '
                         'data.',
                    type=str,
                    dest='data_file')

parser.add_argument('--date-time-now',
                    help='The current date time from which we will compute '
                         'and evaluate the prediction intervals --lead-time '
                         'out. This must be in the format YYYY-MM-DD:HH00 '
                         'using a 24 hour clock.',
                    type=str,
                    dest='date_time_now')

parser.add_argument('--lead-time',
                    help='The number of hours in advance for which the '
                         'prediction interval should be computed.',
                    type=int,
                    dest='lead_time')

parser.add_argument('--hour-range-size',
                    help='The number of contiguous hours to sum up for '
                         'computing prediction intervals.',
                    type=int,
                    dest='hour_range_size')

parser.add_argument('--alpha',
                    help='The significance level of the confidence interval '
                         'you want to compute.',
                    type=float,
                    dest='alpha',
                    default=0.05)

parser.add_argument('--method',
                    help='This specifies which of the two methods for '
                         'computing prediction intervals you want to use.'
                         'If set to 1, this will use fit a vine copula with '
                         'marginals for each of the individual hours and then '
                         'use sums.py to compute the cdf. If set to 2, this '
                         'will sum up the data for contiguous hours and then '
                         'use a univariate distribution for the data.',
                    type=int,
                    dest='method',
                    default=2)

parser.add_argument('--number-of-days',
                    help='The number of days for which you want to evaluate '
                         'the computed intervals. Only required if you run '
                         '"evaluate_compute_intervals.py".',
                    type=int,
                    dest='number_of_days',
                    default=10)

parser.add_argument('--capacity',
                    help='The file that includes the capacities for different'
                         'dates.',
                    type=str,
                    dest='cap')

parser.add_argument('--separation',
                    help='If you want to separate the days by a threshold '
                         'for the forecast, type in "low" for taking just the '
                         'days with a forecast less than the threshold and '
                         '"high" for taking just the days with a forecast '
                         'higher than the threshold. You also have to type '
                         'in a specific threshold.',
                    type=str,
                    dest='sep',
                    default=None)

parser.add_argument('--threshold',
                    help='The threshold for each hour of interest '
                         'for separating the days.',
                    type=float,
                    dest='th',
                    default=None)

parser.add_argument('--window-size',
                    help='The size of the window which you want use for '
                         'segmenting the data.',
                    type = float,
                    dest='ws',
                    default=0.5)

def main():
    #Getting the results of the experiments.
    left_abs_1, right_abs_1, in_abs_1, width_abs_1 , hours_abs_1, \
    length_abs_1 = evaluate_cui.main(0.5)

    left_abs_2, right_abs_2, in_abs_2, width_abs_2 , hours_abs_2, \
    length_abs_2 = evaluate_cui.main(0.25)

    left_rel_1, right_rel_1, in_rel_1, width_rel_1 , hours_rel_1, \
    length_rel_1 = evaluate_percentage.main(0.5)

    left_rel_2, right_rel_2, in_rel_2, width_rel_2 , hours_rel_2, \
    length_rel_2 = evaluate_percentage.main(0.25)

    width_abs_1 = round(width_abs_1, 2)
    width_abs_2 = round(width_abs_2, 2)
    width_rel_1 = round(width_rel_1, 2)
    width_rel_2 = round(width_rel_2, 2)

    hours_abs_1 = round(hours_abs_1, 2)
    hours_abs_2 = round(hours_abs_2, 2)
    hours_rel_1 = round(hours_rel_1, 2)
    hours_rel_2 = round(hours_rel_2, 2)

    args = parser.parse_args()
    future_time = percentage.f_time()


    #Writing the experiments into a .tex file, so that the results can be
    #directly included into a latex document.

    if length_abs_1 == length_abs_2 == length_rel_1 == length_rel_2:
        length = length_rel_1
    else:
        raise ValueError('Error in number of analysis days!')

    date = str(args.date_time_now).replace(':','_')

    if args.sep == 'low':
        sepa = "Low power value characterization"
    elif args.sep == 'high':
        sepa = "High power value characterization"
    else:
        sepa = "Characterization"

    file = open('../research_results/experiments_tex_files/'
                + date + str(args.sep) + '.tex', 'w')
    file.write('\\begin{table}[h] \n'
               '    \\begin{subtable}{.5\\textwidth} \n'
               '        \centering \n'
               '        \\begin{tabular}{|c|c|} \hline \n'
               '            out left & ' + str(left_abs_1) + '\% \\\\ \n'
               '            out right & ' + str(right_abs_1) + '\% \\\\ \n'
               '            inside &' + str(in_abs_1) + '\% \\\\ \n'
               '             avg. width &' + str(width_abs_1) + ' MW \\\\ \hline \n'
               '        \end{tabular} \n'
               '        \caption{absolute error, window size: 0.5, \\\\'
                                'historical hours (avg.): ' + str(hours_abs_1) + '}'
                               '\label{' + date + '_' + str(args.sep) +'_a} \n'
               '    \end{subtable} \n'
               '    \\begin{subtable}{.5\\textwidth} \n'
               '        \centering \n'
               '        \\begin{tabular}{|c|c|} \hline \n'
               '            out left & ' + str(left_rel_1) + '\% \\\\ \n'
               '            out right & ' + str(right_rel_1) + '\% \\\\ \n'
               '            inside & ' + str(in_rel_1) + '\% \\\\ \n'
               '            avg. width & ' + str(width_rel_1) + ' MW \\\\ \hline \n'
               '        \end{tabular} \n'
               '        \caption{relative error, window size: 0.5, \\\\'
                                ' historical hours (avg.): ' + str(hours_rel_1) + '}'
                               '\label{' + date + '_' + str(args.sep) +'_b} \n'
               '    \end{subtable} \n'
               '    \\begin{subtable}{.5\\textwidth} \n'
               '        \centering \n'
               '        \\begin{tabular}{|c|c|} \hline \n'
               '            out left & ' + str(left_abs_2) + '\% \\\\ \n'
               '            out right & ' + str(right_abs_2) + '\% \\\\ \n'
               '            inside & ' + str(in_abs_2) + '\% \\\\ \n'
               '            avg. width & ' + str(width_abs_2) + ' MW \\\\ \hline \n'
               '        \end{tabular} \n'
               '        \caption{absolute error, window size: 0.25, \\\\'
                                ' historical hours (avg.): ' + str(hours_abs_2) + '}'
                               '\label{' + date + '_' + str(args.sep) +'_c} \n'
               '    \end{subtable} \n'
               '    \\begin{subtable}{.5\\textwidth} \n'
               '        \centering \n'
               '        \\begin{tabular}{|c|c|} \hline \n'
               '             out left & ' + str(left_rel_2) + '\% \\\\ \n'
               '            out right & ' + str(right_rel_2) + '\% \\\\ \n'
               '            inside & ' + str(in_rel_2) + '\% \\\\ \n'
               '            avg. width & ' + str(width_rel_2) + ' MW \\\\ \hline \n'
               '        \end{tabular} \n'
               '        \caption{relative error, window size: 0.25, \\\\'
                                ' historical hours (avg.): ' + str(hours_rel_2) + '}'
                               '\label{' + date + '_' + str(args.sep) +'_d} \n'
               '    \end{subtable} \n'
               '    \caption{' + sepa + ' of prediction interval analysis'
                            ' for the first analysis day \mbox{' + str(future_time) +
                            '}, lead time ' + str(args.lead_time)
                            + ' hours, ' + str(length) + ' analysis days, '
                            'historical data beginning at \mbox{' + str(future_time)
                            + '} and $\\alpha=0.3$. The average numbers of'
                              ' historical hours used to compute prediction'
                              ' intervals are located directly below'
                              ' the separate tables.}'
                            '\label{' + date + '_' + str(args.sep) + '} \n'
               '\end{table}')
    file.close()

if __name__ == '__main__':
    main()
