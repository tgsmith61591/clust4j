package com.clust4j.data;

import org.apache.commons.math3.util.FastMath;

/**
 * Split a dataset into a train-test split given a ratio of training data.
 * @author Taylor G Smith
 */
public class TrainTestSplit {
	final private DataSet train;
	final private DataSet test;
	
	
	/**
	 * Split a dataset into a train-test split. Leverages {@link DataSet#shuffle()}
	 * to ensure the most random split possible
	 * @param data
	 * @param train_ratio
	 */
	public TrainTestSplit(DataSet data, double train_ratio) {
		final int m = data.numRows();
		
		// validate the ratio...
		if(train_ratio <= 0.0 || train_ratio >= 1.0) {
			throw new IllegalArgumentException("train ratio must be a positive value between 0.0 and 1.0");
		} else if(m < 2) {
			throw new IllegalArgumentException("too few rows to split");
		}
		
		final int train_rows = FastMath.max((int)FastMath.floor((double)m * train_ratio), 1); // want to make sure at least 1...
		
		// build the split...
		DataSet shuffled = data.shuffle();
		this.train = shuffled.slice(0, train_rows);
		this.test  = shuffled.slice(train_rows, m);
	}
	
	/**
	 * Return a copy of the training set
	 * @return
	 */
	public DataSet getTrain() {
		return train.copy();
	}
	
	/**
	 * Return a copy of the test set
	 * @return
	 */
	public DataSet getTest() {
		return test.copy();
	}
}
