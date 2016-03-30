package com.clust4j.algo;

import com.clust4j.Clust4j;

abstract public class BaseModel extends Clust4j implements java.io.Serializable {
	private static final long serialVersionUID = 4707757741169405063L;
	
	/** The lock to synchronize on for fits */
	final transient Object fitLock = new Object();

	/** This should be synchronized and thread-safe */
	public abstract BaseModel fit();
}
