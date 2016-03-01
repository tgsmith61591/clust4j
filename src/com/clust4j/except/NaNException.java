package com.clust4j.except;

/**
 * Generally thrown in the presence of a {@link Double#NaN}
 * that cannot be handled appropriately.
 * 
 * @author Taylor G Smith
 */
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
