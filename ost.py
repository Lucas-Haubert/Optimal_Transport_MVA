# -*- coding: utf-8 -*-
"""
Original version : @author: Rémi Flamary / GitHub : https://github.com/rflamary/OST/
Modified version : Project Lucas Haubert / Course "Computational Optimal Transport"
"""
import numpy as np


def get_metric(metric,midi_notes,Fs,size_fft,silence_penalty=1e4,epsilon=10,**kwargs):

    """
    Inputs :

    metric : comparison tool
    midi_notes (int) : midi values of notes
    Fs (int) : Sampling frequency of the audio
    size_fft (int) : Size of the Fast Fourier Transform
    silence_penalty (float) : Penalty between midi note equal to 0 and others
    epsilon (int) : Penalty rate on harmonics

    Outputs :

    loss_matrix (2-D array) : Cost matrix C such that c_ij is the square difference of the harmonic-invariant square difference
    fft_freq (1-D array) : Array of fft_frequencies that are compared with pure notes in midi
    
    """
    number_of_notes=len(midi_notes)                                                              # Number of midi notes
    loss_matrix=np.zeros((size_fft//2,number_of_notes))                                          # Only the first half of fft frequencies are relevant (2nd half is its complex conjugate)
    fft_freq=np.fft.fftfreq(size_fft,1.0/Fs)[:size_fft//2]                                       # Array of size_fft//2 elements representing fft frequencies
    f_note=[2.0**((n-60)*1./12)*440 for n in midi_notes]                                         # Array of the frequencies of the notes in midi_notes
    for i in range(number_of_notes):                                                             # For all note (integrer) in midi_notes
        metric_note_i=np.zeros((size_fft//2,))                                                   # 1-D array of size_fft//2 elements
        if metric=='square':                                                                     # Square difference between the frequencies
            metric_note_i=(f_note[i]-fft_freq)**2                                                # metric_note_i = [ (f_note[i]-fft_freq[0])^2  (f_note[i]-fft_freq[1])^2  ...  (f_note[i]-fft_freq[len(m)-1])^2 ]
        elif metric=='harmonic_inv_square':                                                      # Harmonic-invariant transportation cost
            if midi_notes[i]==0:
                metric_note_i[:]=silence_penalty                                                 # metric_note is an array of silence_penalty's (important penalty if a frequency is proposed during a silence)                          
            else:
                q_max=int(fft_freq.max()/f_note[i])                                              # Number of harmonic frequencies to consider
                metric_note_i[:]=np.inf                                                          # Each element of metric_note_i becomes +infinity
                for q in range(1,q_max+1):
                    metric_note_i=np.minimum(metric_note_i,(fft_freq-q*f_note[i])**2+q*epsilon)  # Adapted cost that takes into account harmonics of f_note[i]
        loss_matrix[:,i]=metric_note_i

    return loss_matrix,fft_freq


# Construction of the following couples, example with unmix_plan_fundamental and unmix_fun_fundamental(ind_estim_fund)
# unmix_plan_fundamental(midi_notes,Fs,size_fft) : Get the frequencies of the fundamentals of midi notes in a fft spectrum
# unmix_fun_fundamental(ind_estim_fund) : Builds a function able to isolate these fundamentals in a given frequency spectrum


def unmix_plan_fundamental(midi_notes,Fs,size_fft):
    """
    Inputs :

    midi_notes (int) : midi values of notes
    Fs (int) : Sampling frequency of the audio
    size_fft (int) : Size of the Fast Fourier Transform

    Output :

    ind_estim_fund (1-D array of int) : indexes of the sample nearest from the fundamental for each midi
        those indexes can be used for simple poxer based unmixing

    """
    fft_freq=np.fft.fftfreq(size_fft,1.0/Fs)[:size_fft//2]              # See previous function
    f_note=[2.0**((n-60)*1./12)*440 for n in midi_notes]                # See previous function
    ind_estim_fund = [np.argmin((fn-fft_freq)**2) for fn in f_note]     # 1-D array that represents the indexes in fft spectrum the closest from fundamentals for each midi note      
    
    return ind_estim_fund


def unmix_fun_fundamental(ind_estim_fund):
    """
    Inputs :

    ind_estim_fund (int) : Indexes of the sample (fft frequencies) nearest from the fundamental of each midi

    Output :

    f : Unmixing function for fundamental power
        The goal : isolate notes from a power spectrum (typically obtained from fft)

    """
    size_midi=len(ind_estim_fund)        # Number of midi notes
    def f(v_n,idf=ind_estim_fund):       # v_n : Array that represents a power spectrum (as obtained with fft) ; idf : Indixes of closest sampling (from fft) from the fundamental of each midi
        hat_h_n=np.zeros((size_midi,))   # 1-D array of 0's, same size than midi_notes
        for i in range(len(idf)):        # For all closest fft freq on each midi note
            hat_h_n[i]=v_n[idf[i]]       # Extract the value of spectrum v_n at the fundamental frequency of midi note
        hat_h_n/=hat_h_n.sum()           # Normalize the distribution to 1
        return hat_h_n                       
    
    return f


# The next functions leverage OST approaches, with or without regularization


def unmix_plan_lp(loss_matrix):
    """
    Input :

    loss_matrix (2-D array) : Cost matrix C such that c_ij is the square difference of the harmonic-invariant square difference 
        Computed with : get_metric(metric,midi_notes,Fs,size_fft,**kwargs)

    Output :

    ind_estim_note (1-D array of int) : Index of the note (midi) with minimum cost for each sample (from fft)

    """
    ind_estim_midi = [np.argmin(loss_matrix[i,:]) for i in range(loss_matrix.shape[0])]

    return ind_estim_midi


def unmix_fun_lp(ind_estim_midi,number_fft):
    """
    Inputs :

    ind_estim_midi (int) : Index of the note (midi) with minimum cost for each sample (from fft)
    number_fft (int) : Number of fft frequencies

    Output :

    f : Unmixing function for classical OST approach
    
    """
    def f(v_n,idf=ind_estim_midi,number_fft=number_fft):       # v_n : Array that represents a spectrum (as obtained with fft) ; number_fft : Number of fft frequencies
        hat_h_n=np.zeros((number_fft,))                        # 1-D array of size number_fft
        for i in range(len(idf)):                              # For i (= fft frequency) index of ind_estim_midi
            hat_h_n[idf[i]]+=v_n[i]                            # idf[i] is a midi integer note representation
        return hat_h_n                                         # hat_h_n contains a cumulated repartition od fft values
    return f


# Group regularization

def unmix_fun_lp_sparse(loss_matrix,mu=0.5,nb_iter=2,eps=1e-6,**kwargs):
    """
    Inputs :

    loss_matrix (2-D array) : Cost matrix C such that c_ij is the square difference of the harmonic-invariant square difference
    mu (float) : Parameter for R_iter
    nb_iter (int) : Number of iterations in the process 
    eps (float) : Avoid divin+sion by 0

    Output :

    f : Unmixing function for sparse OST approach

    """
    size_midi=loss_matrix.shape[1]                             # Number of MIDI notes
    def f(v_n,C=loss_matrix,mu=mu,nb_iter=nb_iter,eps=eps):
        R_iter=np.zeros((1,size_midi))
        for it in range(nb_iter):
            hat_h_n=np.zeros((size_midi,))
            C_iter=C+R_iter
            for i in range(C.shape[0]):
                isel=np.argmin(C_iter[i,:])
                hat_h_n[isel]+=v_n[i]
            R_iter=mu*1.0/(np.sqrt(hat_h_n)+eps)
        return hat_h_n

    return f


def unmix_plan_entrop(loss_matrix,lambda_e):
    """
    returns the rpe compute L plan for entropic regularized OST

    Inputs :

    loss_matrix (2-D array) : Cost matrix C such that c_ij is the square difference of the harmonic-invariant square difference
    lambda_e (float) : Entropic regularization factor

    Output :

    L_e (2-D array) : Matrix to compute hat_h_n in unmix_fun_entrop(L_e)

    """
    E=np.exp(-loss_matrix/lambda_e/loss_matrix.max())
    L_e = E*1./(E.sum(1).reshape((loss_matrix.shape[0],1)))
    return L_e


def unmix_fun_entrop(L_e):
    """
    returns the unmixing function for entropic OST unmixing 

    Inputs :

    L_e (2-D array) : Array to compute hat_h_n from v_n

    Output :

    f : Unmixing function for entropic OST approach

    """    
    def f(v_n,L_e=L_e):
        hat_h_n = L_e.T.dot(v_n)
        return hat_h_n
    
    return f


def get_unmix_fun(midi_notes,Fs,size_fft,method='fund',metric='harmonic_inv_square',lambda_e=1e-3,**kwargs):
    """
    Inputs :

    midi_notes (int) : midi values of notes
    Fs (int) : Sampling frequency of the audio
    size_fft (int) : Size of the Fast Fourier Transform
    method (str) : Unmixing approach
    metric : comparison tool

    Output :

    f : Unmixing function for the given method

    """
    if method.lower()=='fund':
        ind_estim_fund=unmix_plan_fundamental(midi_notes,Fs,size_fft)
        f=unmix_fun_fundamental(ind_estim_fund)

    elif method.lower() in ['lp','ost']:
        loss_matrix,_=get_metric(metric,midi_notes,Fs,size_fft,**kwargs)
        ind_estim_midi=unmix_plan_lp(loss_matrix)
        f=unmix_fun_lp(ind_estim_midi,size_fft//2)

    elif method.lower() in ['oste','entrop']:
        loss_matrix,_=get_metric(metric,midi_notes,Fs,size_fft,**kwargs)
        L_e=unmix_plan_entrop(loss_matrix,lambda_e)
        f=unmix_fun_entrop(L_e)

    elif method.lower() in ['lp_sparse','ost_sparse','ostg']:
        loss_matrix,_=get_metric(metric,midi_notes,Fs,size_fft,**kwargs)
        f=unmix_fun_lp_sparse(loss_matrix,**kwargs)

    return f
