import { Stack, usePathname } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { useLayoutEffect } from "react";
import { Platform, StyleSheet, View } from "react-native";
import StudioDOM from "@/components/dom/StudioDOM";
import { colors } from "@/theme/colors";
import { getApiBase } from "./index";

function isStudioWebRoute(path: string): boolean {
  return (
    path === "/" ||
    path === "/piano" ||
    path === "/projects" ||
    path === "/collections" ||
    path === "/songscription" ||
    path.startsWith("/songscription/") ||
    path.startsWith("/p/")
  );
}

export default function RootLayout() {
  const path = usePathname();
  const showWebStudio = Platform.OS === "web" && isStudioWebRoute(path);

  useLayoutEffect(() => {
    if (Platform.OS !== "web") return;
    const section = path === "/" ? "New transcription"
      : path === "/piano" ? "Piano"
      : path.startsWith("/songscription") ? "Songscription"
      : path === "/projects" ? "Projects"
      : path === "/collections" ? "Collections"
      : path.startsWith("/p/") ? "Studio"
      : path === "/auth" ? "Login"
      : "Harmony Auralis";
    document.title = `${section} - Harmony Auralis`;
  }, [path]);

  return (
    <View style={styles.shell}>
      <StatusBar style="dark" />
      {Platform.OS === "web" && (
        <View
          style={showWebStudio ? styles.studioVisible : styles.studioBackground}
          pointerEvents={showWebStudio ? "auto" : "none"}
        >
          <StudioDOM apiBase={getApiBase()} path={path} dom={{ style: { flex: 1 }, scrollEnabled: false }} />
        </View>
      )}
      <View style={showWebStudio ? styles.hiddenStack : styles.stackVisible}>
        <Stack screenOptions={{ headerShown: false, contentStyle: { backgroundColor: colors.background } }} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  shell: { flex: 1, backgroundColor: colors.background },
  studioVisible: { flex: 1, width: "100%", height: "100%" },
  studioBackground: {
    position: "absolute",
    right: 0,
    bottom: 0,
    width: 320,
    height: 180,
    opacity: 0.001,
    zIndex: -1,
    overflow: "hidden",
  },
  stackVisible: { flex: 1 },
  hiddenStack: { display: "none" },
});
