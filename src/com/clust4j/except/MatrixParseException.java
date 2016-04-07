/*******************************************************************************
 *    Copyright 2015, 2016 Taylor G Smith
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *******************************************************************************/
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
