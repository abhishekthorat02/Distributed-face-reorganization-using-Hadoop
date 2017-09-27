import java.io.*;


public class JavaRunCommand {


	public static void main(String args[]) throws InterruptedException {
		
		try{
			Process ffmpeg = Runtime.getRuntime().exec("avconv -i /tmp/demo.mp4 -r 1 -f image2 /home/hduser/Standalone/face_crop/testPics/image-%3d.jpg");
			ffmpeg.waitFor();
			Process face_crop = Runtime.getRuntime().exec("python facecrop.py ", null, new File("/home/hduser/Standalone/face_crop/"));
			face_crop.waitFor();
			ProcessBuilder builder = new ProcessBuilder("python", "create_csv.py");
			builder.directory(new File("/home/hduser/Standalone/create_csv/"));
			builder.redirectOutput(new File("/home/hduser/Standalone/create_csv/output/out.txt"));
			builder.redirectError(new File("/home/hduser/Standalone/create_csv/output/outerror.txt"));
			Process p = builder.start(); // throws IOException
			p.waitFor();
			try (PrintWriter outline = new PrintWriter(new BufferedWriter(new FileWriter("/home/hduser/Standalone/create_csv/output/out.txt", true)))) {
				outline.println("/tmp/reference.jpg;9999999");
			} catch (IOException e) {
				//exception handling left as an exercise for the reader
			}
			try (PrintWriter outpref = new PrintWriter(new BufferedWriter(new FileWriter("/home/hduser/Standalone/create_csv/output/out.txt", true)))) {} catch (IOException e) {
				System.out.println("exception");
				e.printStackTrace();
				System.exit(-1);
			}
			System.setOut(new PrintStream(new FileOutputStream("/home/hduser/Standalone/output/outsingle.txt")));
			try {
				String line;

				Process proc = Runtime.getRuntime().exec("./face /home/hduser/Standalone/create_csv/output/out.txt 50.0", null, new File("/home/hduser/Standalone/face_rec/"));
				BufferedRezader in = new BufferedReader(
				new InputStreamReader(proc.getInputStream()));
				while ((line = in.readLine()) != null) {
					System.out.println(line);
				} in .close();
			} catch (Exception e) {
				// ...
			}
		}catch(IOException e){
		}
		//System.out.println(folderref);
	}
}

