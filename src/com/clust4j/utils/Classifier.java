package com.clust4j.utils;

/**
 * An interface for classifiers, both supervised and unsupervised.
 * @author Taylor G Smith
 */
public interface Classifier extends java.io.Serializable {
	/**
	 * Returns a copy of the assigned class labels in
	 * record order
	 * @return
	 */
	public int[] getLabels();
}
