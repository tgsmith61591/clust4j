package com.clust4j;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import lombok.NonNull;

/**
 * The absolute super type for all clust4j objects (models and datasets)
 * that should be able to commonly serialize their data.
 * @author Taylor G Smith
 */
public abstract class Clust4j implements java.io.Serializable {
	private static final long serialVersionUID = -4522135376738501625L;
	
	/**
	 * Load a model from a FileInputStream
	 * @param fos
	 * @return
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public static Clust4j loadObject(@NonNull final FileInputStream fis) throws IOException, ClassNotFoundException {
		ObjectInputStream in = null;
		Clust4j bm = null;
			
		try {
			in = new ObjectInputStream(fis);
	        bm = (Clust4j) in.readObject();
		} finally {
			if(null != in)
				in.close();

			fis.close();
		}
        
        return bm;
	}
	
	/**
	 * Save a model to FileOutputStream
	 * @param fos
	 * @throws IOException
	 */
	public void saveObject(@NonNull final FileOutputStream fos) throws IOException {
		ObjectOutputStream out = null;
		
		try {
			out = new ObjectOutputStream(fos);
			out.writeObject(this);
		} finally {
			if(null != out)
				out.close();

			fos.close();
		}
	}
}
