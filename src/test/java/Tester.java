import agi.nn.playground.Playground;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

@Slf4j
public class Tester {
    @Test
    public void test() throws Exception {
        new Playground().batch(10);
    }
}
