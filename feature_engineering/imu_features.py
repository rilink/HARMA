import os
import numpy as np
import pandas as pd
from scipy.stats import iqr, entropy
from scipy.fftpack import fft
from scipy.stats import entropy
from scipy.signal import find_peaks
import ast
from scipy.stats import iqr, kurtosis, skew
from scipy.signal import welch
import math
from collections import Counter


def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError, TypeError):
        return [] 

def signal_magnitude_vector(window):
    magnitude = np.sqrt(np.sum(np.square(window), axis=1)) 
    return magnitude

def spectral_entropy(signal, sf=50):
    freqs, psd = welch(signal, sf)
    psd_norm = psd / np.sum(psd)
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))  
    return entropy

def hjorth_parameters(signal):
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)
    var_zero = np.var(signal)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)
    mobility = np.sqrt(var_d1 / var_zero) if var_zero != 0 else 0
    complexity = np.sqrt(var_d2 / var_d1) / mobility if var_d1 != 0 else 0
    return mobility, complexity

def mean_crossing_rate(signal):
    mean_val = np.mean(signal)
    crossings = np.where(np.diff(np.signbit(signal - mean_val)))[0]
    return len(crossings) / len(signal)

def differential_entropy(signal):
    var = np.var(signal)
    return 0.5 * np.log(2 * np.pi * np.e * var)


#https://github.com/raphaelvallat/entropy/blob/master/entropy/fractal.py
def petrosian_fd(signal):
    diff = np.diff(signal)
    N_delta = np.sum(diff[1:] * diff[:-1] < 0)
    n = len(signal)
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * N_delta)))

def katz_fd(signal):
    L = np.sum(np.sqrt(np.square(np.diff(signal))))
    d = np.max(np.abs(signal - signal[0]))
    n = len(signal)
    return np.log10(n) / (np.log10(d / L + 1e-6))  

def orientation(x, y, z, dt=0.02):     
    acc_data = np.stack([x, y, z], axis=1)
    velocity = np.cumsum(acc_data * dt, axis=0)
    position = np.cumsum(velocity * dt, axis=0)
    pos_x = position[-1][0]
    pos_y = position[-1][1]
    pos_z = position[-1][2]
    velo_acc = np.sum(velocity[-1])
    pos_acc = np.sum(position[-1])
    return pos_x, pos_y, pos_z, velo_acc, pos_acc

def calculate_1D_features(signal, features, axis_name):
    result = {}    
    signal = np.array(signal)

    # time-domain features
    if 'mean' in features:
        result[f'{axis_name}_mean'] = np.mean(signal)
    if 'var' in features:
        result[f'{axis_name}_var'] = np.var(signal)
    if 'std' in features:
        result[f'{axis_name}_std'] = np.std(signal)
    if 'rms' in features:
        result[f'{axis_name}_rms'] = np.sqrt(np.mean(signal**2))
    if 'shape_factor' in features: 
        mv = np.mean(signal)
        rms = np.sqrt(np.mean(signal**2))
        shape_factor = rms / np.abs(mv)
        result[f'{axis_name}_shape_factor'] = shape_factor

    if 'min' in features:
        result[f'{axis_name}_min'] = np.min(signal)
    if 'max' in features:
        result[f'{axis_name}_max'] = np.max(signal)
    if 'median' in features:
        result[f'{axis_name}_median'] = np.median(signal)
    if 'iqr' in features:
        result[f'{axis_name}_iqr'] = iqr(signal)
    if 'sma' in features:
        result[f'{axis_name}_sma'] = np.sum(np.abs(signal)) / len(signal)
    if 'skewness' in features:
        result[f'{axis_name}_skewness'] = skew(signal)
    if 'kurtosis' in features:
        result[f'{axis_name}_kurtosis'] = kurtosis(signal)
    if 'jerk_mean' in features: 
        j = np.diff(signal)
        result[f'{axis_name}_jerk_mean'] = np.mean(np.abs(j))
    if 'jerk_std' in features:
        j = np.diff(signal)
        result[f'{axis_name}_jerk_std'] = np.std(j)
    if 'num_peak' in features: 
        peaks, _ = find_peaks(signal, distance=3)
        result[f'{axis_name}_peaks'] = len(peaks)
    if 'zero_crossings' in features: 
        result[f'{axis_name}_zero_crossings'] = np.sum((signal[:-1] * signal[1:]) < 0)

    # frequency-domain features
    fft_vals = np.abs(fft(signal))
    fft_vals = fft_vals[:len(fft_vals)//2]  
    if 'energy' in features:
        result[f'{axis_name}_energy'] = np.sum(fft_vals ** 2)
    if 'dominant_freq' in features:
        result[f'{axis_name}_dominant_freq'] = np.argmax(fft_vals)
    if 'entropy' in features:
        prob_dist = fft_vals / np.sum(fft_vals)
        result[f'{axis_name}_entropy'] = entropy(prob_dist)
    if 'signal_entropy' in features: 
        hist, _ = np.histogram(signal, bins=10, density=True)
        result[f'{axis_name}_signal_entropy'] = entropy(hist + 1e-10)  
    return result

def calculate_3D_features(row, features):
    """Features computed jointly across all three accelerometer axes."""
    result = {}
    x = np.asarray(row['x_axis'])
    y = np.asarray(row['y_axis'])
    z = np.asarray(row['z_axis'])
    smv = signal_magnitude_vector(np.column_stack((x, y, z)))
    x_mean, y_mean, z_mean = np.mean(x), np.mean(y), np.mean(z)

    if 'pos_x' or 'pos_y' or 'pos_z' or 'velo_acc' or 'pos_acc' in features:
        pos_x, pos_y, pos_z, velo_acc, pos_acc = orientation(x,y,z, dt=0.02)
        if 'pos_x' in features: 
            result['pos_x'] = pos_x
        if 'pos_y' in features: 
            result['pos_y'] = pos_y
        if 'pos_z' in features: 
            result['pos_z'] = pos_z
        if 'velo_acc' in features: 
            result['velo_acc'] = velo_acc
        if 'pos_acc' in features: 
            result['pos_acc'] = pos_acc

    if 'pitch' in features:
        pitch = np.degrees(np.arctan2(-x_mean, np.sqrt(y_mean**2 + z_mean**2)))
        result['pitch'] = pitch
    if 'roll' in features:
        roll = np.degrees(np.arctan2(y_mean, z_mean))
        result['roll'] = roll
    if 'total_sqrt' in features:
        result['total_sqrt'] = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2) 
    if 'total' in features:
        result['total'] = x_mean + y_mean + z_mean
    if 'signal_magnitude' in features:
        result['signal_magnitude'] = np.sum((x) + (y) + (z)) / len(x)
    if 'signal_magnitude_abs' in features:
        result['signal_magnitude_abs'] = np.sum(np.abs(x) + np.abs(y) + np.abs(z)) / len(x)

    if 'axis_correlation' in features:
        result.update({
            'axis_correlation_x_y': np.corrcoef(x, y)[0, 1],
            'axis_correlation_x_z': np.corrcoef(x, z)[0, 1],
            'axis_correlation_y_z': np.corrcoef(y, z)[0, 1],
        })

    if 'axis_covariance' in features:
        result.update({
            'axis_covariance_x_y': np.cov(x, y)[0, 1],
            'axis_covariance_x_z': np.cov(x, z)[0, 1],
            'axis_covariance_y_z': np.cov(y, z)[0, 1],
        })

    if 'tilt_angle' in features:
        norm = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2) + 1e-6
        result['tilt_angle'] = np.degrees(np.arccos(z_mean / norm))

    if 'spectral_entropy' in features:
        result['spectral_entropy'] = spectral_entropy(smv)
    if 'min_smv' in features:
        result['min_smv'] = np.min(smv)
    if 'max_smv' in features:
        result['max_smv'] = np.max(smv)
    if 'ptp_smv' in features:
        result['ptp_smv'] = np.ptp(smv)
    if 'iqr_smv' in features:
        result['iqr_smv'] = iqr(smv)
    if 'std_smv' in features:
        result['std_smv'] = np.std(smv)
    if 'skew_smv' in features:
        result['skew_smv'] = skew(smv)
    if 'kurtosis_smv' in features:
        result['kurtosis_smv'] = kurtosis(smv)
    if 'mobility' in features or 'complexity' in features:
        mobility, complexity = hjorth_parameters(smv)
        if 'mobility' in features:
            result['mobility'] = mobility
        if 'complexity' in features:
            result['complexity'] = complexity
    if 'mean_crossing' in features:
        result['mean_crossing'] = mean_crossing_rate(smv)
    if 'differential_entropy' in features:
        result['differential_entropy'] = differential_entropy(smv)
    if 'petrosian_fd' in features:
        result['petrosian_fd'] = petrosian_fd(smv)
    if 'katz_fd' in features:
        result['katz_fd'] = katz_fd(smv)
    return result

################################################################################################################################################

def feature_calculation(
    df, #raw sensor data
    output_path='processed_data\\one_sec_data_features.csv',
    features_1D=['mean', 'var', 
                 'rms', 'shape_factor', # new features
                 'min', 'max', 'median', 'iqr', 'sma', 'energy', 'dominant_freq', 
                 'entropy', 'signal_entropy', 'jerk_mean', 'jerk_std', 'num_peak', 'zero_crossings'
                 ],
    features_3D=[
        'pos_x', 'pos_y', 'pos_z', 'velo_acc', 'pos_acc', # trajectory features
        'pitch', 'roll', #orienation features
        'total', 'total_sqrt', 'signal_magnitude', 'signal_magnitude_abs', 
        'axis_correlation', 'axis_covariance', 'tilt_angle',
        'spectral_entropy', 'min_smv', 'max_smv', 'ptp_smv', 'iqr_smv', 'std_smv', 'skew_smv',
        'kurtosis_smv', 'mobility', 'complexity', 'mean_crossing', 'differential_entropy', 'petrosian_fd', 'katz_fd'
        ],
    add_features=None, # if provided the new features will be calculated (not the old ones recalculated)
    remove_features=None, # if provided the features to be removed will be removed (not the old ones recalculated)
    update_existing=False
):

    existing_df = None
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)

    if (remove_features is not None) and (add_features is None):
        if existing_df is None:
            return pd.DataFrame(columns=[])
        to_drop = []
        for c in remove_features: 
            if c in features_1D:
                for axis in ['x', 'y', 'z']:
                    to_drop.append(f"{axis}_{c}")
            elif c in features_3D: 
                to_drop.append(c)

        final_df = existing_df.drop(columns=to_drop)
        final_df.to_csv(output_path, index=False)
        print('Features removed')
        return final_df

    for axis in ['x_axis', 'y_axis', 'z_axis']:
        df[axis] = df[axis].apply(safe_literal_eval)

    feature_rows = []
    for idx, row in df.iterrows():
        if all(row[axis] != [] for axis in ['x_axis', 'y_axis', 'z_axis']):
            base_info = row.drop(['x_axis', 'y_axis', 'z_axis']).to_dict()
            existing_row = (
                existing_df[existing_df['id'] == row['id']].iloc[0]
                if existing_df is not None and row['id'] in existing_df['id'].values
                else None
            )
            new_features = {}

            for axis in ['x_axis', 'y_axis', 'z_axis']:
                if row[axis] == []:
                    continue

                axis_code = axis[0] 
                all_1d_fts = features_1D

                if add_features is not None:
                    requested_1d_fts = [f for f in features_1D if f in add_features]
                else:
                    requested_1d_fts = all_1d_fts

                if requested_1d_fts:
                    stats = calculate_1D_features(row[axis], requested_1d_fts, axis_name=axis_code)

                    for col_name, col_val in stats.items():
                        if existing_df is not None and (not update_existing) and (col_name in existing_df.columns):
                            new_features[col_name] = existing_row[col_name]
                        else:
                            new_features[col_name] = col_val

            if add_features is not None:
                requested_3d_fts = [f for f in features_3D if f in add_features]
            else:
                requested_3d_fts = features_3D

            if requested_3d_fts:
                combined_3d_list = requested_3d_fts
                dim3_stats = calculate_3D_features(row, combined_3d_list)

                for col_name, col_val in dim3_stats.items():
                    if existing_df is not None and (not update_existing) and (col_name in existing_df.columns):
                        new_features[col_name] = existing_row[col_name]
                    else:
                        new_features[col_name] = col_val

            base_info.update(new_features)
            feature_rows.append(base_info)

    final_df = pd.DataFrame(feature_rows)

    if existing_df is not None:
        # any column that was in existing_df but not in final_df should be re-appended unchanged
        missing_cols = [c for c in existing_df.columns if c not in final_df.columns]
        for c in missing_cols:
            final_df[c] = existing_df[c]

    if remove_features:
        # only drop those actually present
        to_drop = [c for c in remove_features if c in final_df.columns]
        final_df = final_df.drop(columns=to_drop)

    final_df.to_csv(output_path, index=False)
    return final_df

if __name__ == "__main__":
    # Not part of the reproduction pipeline: build_feature_tables.py imports
    # feature_calculation() directly and computes its own windows from raw data.
    # This standalone entry point requires archive/imu_only/preprocessing.py's
    # output, which the documented pipeline no longer generates.
    from pathlib import Path
    _PROCESSED = Path(__file__).resolve().parent / "processed_data"
    _PROCESSED.mkdir(parents=True, exist_ok=True)
    one_sec_data = pd.read_csv(_PROCESSED / "one_sec_data_overlap_threshold.csv")
    feature_calculation(one_sec_data, output_path=str(_PROCESSED / "one_sec_data_features.csv"))


    # one_sec_data = pd.read_csv('..\..\..\data\processed_data\one_sec_data_threshold.csv')
    # one_sec_data_with_features = feature_calculation(one_sec_data, output_path='processed_data\\one_sec_data_features_threshold_new.csv', remove_features=['jerk_std'])

#removed: total, median, 'pos_x', 'pos_y', 'pos_z', 'velo_acc', 'pos_acc', 'ptp_smv', 'tilt_angle', 'jerk_std'

    # one_sec_data_with_features = feature_calculation(
    #     one_sec_data, #raw sensor data
    #     output_path='processed_data\\one_sec_data_features_threshold_new.csv',
    #     features_1D=['mean', 'var', 
    #                 'rms', 'shape_factor', # new features
    #                 'min', 'max', 'median', 'iqr', 'sma', 'energy', 'dominant_freq', 
    #                 'entropy', 'signal_entropy', 'jerk_mean', 'jerk_std', 'num_peak', 'zero_crossings'
    #                 ],
    #     features_3D=[
    #         'pos_x', 'pos_y', 'pos_z', 'velo_acc', 'pos_acc', # trajectory features
    #         'pitch', 'roll', #orienation features
    #         'total', 'total_sqrt', 'signal_magnitude', 'signal_magnitude_abs', 
    #         'axis_correlation', 'axis_covariance', 'tilt_angle',
    #         'spectral_entropy', 'min_smv', 'max_smv', 'ptp_smv', 'iqr_smv', 'std_smv', 'skew_smv',
    #         'kurtosis_smv', 'mobility', 'complexity', 'mean_crossing', 'differential_entropy', 'petrosian_fd', 'katz_fd'
    #         ],
    #     add_features=None, # if provided the new features will be calculated (not the old ones recalculated)
    #     remove_features=None, # if provided the features to be removed will be removed (not the old ones recalculated)
    #     update_existing=False
    # )

    # one_sec_data = pd.read_csv('..\..\..\data\processed_data\one_sec_data_overlap_threshold_jitter.csv')
    # one_sec_data_with_features = feature_calculation(one_sec_data, output_path='processed_data\\one_sec_data_features_overlap_threshold_jitter.csv')
    #one_sec_data_with_features = feature_calculation(one_sec_data, remove_features=['min_smv'])
    #one_sec_data_with_features = feature_calculation(one_sec_data, add_features=['min_smv'])
    #one_sec_data_with_features = feature_calculation(one_sec_data, remove_features=['mean'])
    #one_sec_data_with_features = feature_calculation(one_sec_data, add_features=['mean'])
    # one_sec_data = pd.read_csv('..\..\..\data\processed_data\one_sec_data_threshold.csv')
    # one_sec_data_with_features = feature_calculation(one_sec_data, output_path='processed_data\\one_sec_data_features_threshold_recalc.csv', remove_features=['signal_magnitude_abs', 'signal_magnitude', 'total_sqrt', 'total', 'pos_x', 'pos_y', 'pos_z', 'velo_acc', 'pos_acc'])
    # one_sec_data = pd.read_csv('..\..\..\data\processed_data\one_sec_data_threshold.csv')
    # one_sec_data_with_features = feature_calculation(one_sec_data, output_path='processed_data\\one_sec_data_features_threshold_recalc.csv', add_features=['signal_magnitude_abs', 'signal_magnitude', 'total_sqrt', 'total', 'pos_x', 'pos_y', 'pos_z', 'velo_acc', 'pos_acc'])
    
    
    # orig = pd.read_csv('processed_data\\one_sec_data_features_threshold.csv')
    # recalc = pd.read_csv('processed_data\\one_sec_data_features_threshold_recalc.csv')
    # for element in orig.columns:
    # if element not in ['label','sensor_location'] and element in recalc.columns:
    #     print(element)
    #     # Compare elements, count how many differ by >= 0.001
    #     diff = np.abs(orig[element] - recalc[element]) >= 0.001
    #     print(np.sum(diff))