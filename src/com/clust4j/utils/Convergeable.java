package com.clust4j.utils;

public interface Convergeable {
	public boolean didConverge();
	public int getMaxIter();
	public double getMinChange();
	public int itersElapsed();
}
