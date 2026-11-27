import Constants from "expo-constants";
import { usePathname } from "expo-router";
import { Platform, StyleSheet, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import StudioDOM from "@/components/dom/StudioDOM";
import { colors } from "@/theme/colors";

export function getApiBase() {
  const configured = process.env.EXPO_PUBLIC_API_URL?.trim();
  if (configured) return configured.replace(/\/$/, "");

  if (Platform.OS === "web") {
    const host = typeof window === "undefined" ? "127.0.0.1" : window.location.hostname;
    return `http://${host}:8100`;
  }

  const host = Constants.expoConfig?.hostUri?.split(":")[0];
  if (host) return `http://${host}:8100`;
  return Platform.OS === "android" ? "http://10.0.2.2:8100" : "http://127.0.0.1:8100";
}

export default function HomeScreen() {
  const path = usePathname();
  // On web, RootLayout mounts a single persistent StudioDOM so audio/video playback never stops across route changes.
  if (Platform.OS === "web") return null;
  return (
    <SafeAreaView edges={["top", "bottom"]} style={styles.screen}>
      <View style={styles.content}>
        <StudioDOM apiBase={getApiBase()} path={path} dom={{ style: { flex: 1 }, scrollEnabled: false }} />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background },
  content: { flex: 1 },
});
