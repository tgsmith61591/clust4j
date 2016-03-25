package com.clust4j.except;

public class MatrixParseException extends RuntimeException {
	private static final long serialVersionUID = 5494488803473338495L;

	public MatrixParseException() {
		super();
	}
	
	public MatrixParseException(String msg) {
		super(msg);
	}
	
	public MatrixParseException(Throwable cause) {
		super(cause);
	}
	
	public MatrixParseException(String msg, Throwable cause) {
		super(msg, cause);
	}
}
