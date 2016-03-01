package com.clust4j.except;

public class ModelNotFitException extends RuntimeException {

	private static final long serialVersionUID = -7868815497000388833L;

	public ModelNotFitException() {
		super();
	}
	
	public ModelNotFitException(final String msg) {
		super(msg);
	}
	
	public ModelNotFitException(final Throwable thrown) {
		super(thrown);
	}
	
	public ModelNotFitException(final String msg, final Throwable thrown) {
		super(msg, thrown);
	}
}
