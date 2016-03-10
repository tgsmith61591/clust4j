package com.clust4j.algo;

/**
 * An extension of the {@link SafeLabelEncoder} that accounts for the noisey
 * labels that {@link NoiseyClusterer} algorithms can produce.
 * 
 * Since noisey clusterers have a propensity to create
 * predictions of only -1, we need to allow single mappings
 * in this subclass. This avoids the IllegalArgumentException
 * in the super class constructor.
 * @author Taylor G Smith
 */
public class NoiseyLabelEncoder extends SafeLabelEncoder {
	private static final long serialVersionUID = -5898357662470826812L;
	public static final int NOISE_CLASS = NoiseyClusterer.NOISE_CLASS;

	public NoiseyLabelEncoder(int[] labels) {
		super(labels);
		addMapping(NOISE_CLASS, NOISE_CLASS);
	}
	
	@Override
	public NoiseyLabelEncoder fit() {
		synchronized(this) {
			return (NoiseyLabelEncoder) super.fit();
		}
	}
}
