package com.clust4j.utils;

public class NaNException extends RuntimeException {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3297235577826195591L;

	public NaNException() {
		super();
	}
	
	public NaNException(final String msg) {
		super(msg);
	}
	
	public NaNException(Throwable thrown) {
		super(thrown);
	}
	
	public NaNException(String msg, Throwable thrown) {
		super(msg, thrown);
	}
}
