package com.clust4j.except;

import java.io.IOException;

public class MatrixParseException extends IOException {
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
