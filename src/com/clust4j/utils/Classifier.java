package com.clust4j.utils;

/**
 * An interface for classifiers, both supervised and unsupervised.
 * @author Taylor G Smith
 */
public interface Classifier {
	/**
	 * Returns the ground truth labels in a 
	 * supervised context, the identified labels in
	 * an unsupervised context.
	 * @return
	 */
	public int[] getLabels();
}
